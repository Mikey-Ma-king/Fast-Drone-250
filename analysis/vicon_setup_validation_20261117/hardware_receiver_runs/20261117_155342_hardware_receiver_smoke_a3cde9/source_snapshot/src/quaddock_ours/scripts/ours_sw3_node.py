#!/usr/bin/env python3
"""ROS adapter for the frozen Ours-SW3 estimator.

The node consumes the existing Fast-Drone-250 ``nav_msgs/Odometry`` streams,
keeps the estimator causal, and publishes only an estimated target state.  It
does not arm, take off, land, or publish flight-control commands.
"""
from __future__ import division

import bisect
import copy
import hashlib
import io
import json
import math
import threading
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

try:
    from tf.transformations import quaternion_matrix, quaternion_from_euler
except ImportError:  # pragma: no cover - only used on a ROS host
    quaternion_matrix = None
    quaternion_from_euler = None

from quaddock_ours.estimator import make_estimator
from quaddock_ours.models import drz, rz, uav_rotation, wrap


def _finite_vector(values, size):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("expected a finite vector of length %d" % size)
    return result


def _stamp(msg):
    stamp = msg.header.stamp.to_sec()
    return float(stamp if stamp > 0.0 else rospy.Time.now().to_sec())


def _quat_xyzw(msg):
    q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    q = _finite_vector(q, 4)
    if np.linalg.norm(q) < 1e-9:
        return None
    return (q / np.linalg.norm(q)).tolist()


def _yaw_from_odom(msg, mode):
    if mode == "legacy_yaw_w":
        return float(msg.pose.pose.orientation.w)
    q = _quat_xyzw(msg)
    if q is None:
        return 0.0
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw):
    if quaternion_from_euler is not None:
        return quaternion_from_euler(0.0, 0.0, float(yaw))
    return (0.0, 0.0, math.sin(float(yaw) / 2.0), math.cos(float(yaw) / 2.0))


class OdomHistory(object):
    """Small received-only history used to causally match relative messages."""

    def __init__(self, max_length=400):
        self.max_length = int(max_length)
        self.times = []
        self.values = []

    def add(self, stamp, value):
        i = bisect.bisect_left(self.times, stamp)
        if i < len(self.times) and abs(self.times[i] - stamp) < 1e-9:
            self.values[i] = value
            return False
        self.times.insert(i, float(stamp))
        self.values.insert(i, value)
        if len(self.times) > self.max_length:
            del self.times[0]
            del self.values[0]
        return True

    def latest(self):
        return self.values[-1] if self.values else None

    def match(self, stamp, max_gap):
        if not self.times:
            return None
        i = bisect.bisect_left(self.times, stamp)
        if i < len(self.times) and abs(self.times[i] - stamp) < 1e-8:
            return dict(self.values[i])
        if i == 0 or i == len(self.times):
            return None
        t0, t1 = self.times[i - 1], self.times[i]
        if t1 - t0 > float(max_gap):
            return None
        a, b = self.values[i - 1], self.values[i]
        alpha = (float(stamp) - t0) / (t1 - t0)
        value = dict(a)
        for key in ("position", "velocity"):
            value[key] = ((1.0 - alpha) * np.asarray(a[key]) +
                          alpha * np.asarray(b[key])).tolist()
        dyaw = float(wrap(float(b["yaw"]) - float(a["yaw"])))
        value["yaw"] = float(a["yaw"] + alpha * dyaw)
        if a.get("quaternion_xyzw") is not None and b.get("quaternion_xyzw") is not None:
            qa = np.asarray(a["quaternion_xyzw"], dtype=float)
            qb = np.asarray(b["quaternion_xyzw"], dtype=float)
            if float(qa.dot(qb)) < 0.0:
                qb = -qb
            q = (1.0 - alpha) * qa + alpha * qb
            q /= max(np.linalg.norm(q), 1e-12)
            value["quaternion_xyzw"] = q.tolist()
        value["interpolated_from"] = [t0, t1]
        return value


class OursSW3Node(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(rospy.get_param("~estimator", {}))
        self.output_hz = float(rospy.get_param("~output_hz", 50.0))
        self.max_odom_gap = float(rospy.get_param("~max_odom_gap_s", 0.10))
        # Mirror the original 100-ms status check: five missed checks mark a
        # stream unavailable. Readiness is set after causal vision startup.
        self.aoa_received = False
        self.aoa_seq = 0
        self.last_aoa_seq = 0
        self.last_aoa_arrival = None
        self.aoa_status_tick = None
        self.uav_received = False
        self.last_uav_arrival = None
        self.uav_status_tick = None
        self.match_wait_s = float(rospy.get_param("~match_wait_s", 0.05))
        self.pending = []
        self.received_relative = set()
        self.dog_yaw_mode = str(rospy.get_param("~dog_yaw_mode", "legacy_yaw_w"))
        self.vision_yaw_mode = str(rospy.get_param("~vision_yaw_mode", "legacy_yaw_w"))
        self.vision_input_mode = str(rospy.get_param("~vision_input_mode", "world_odom"))
        self.uav_topic = str(rospy.get_param("~uav_odom_topic", "/vins_fusion/imu_propagate"))
        self.dog_topic = str(rospy.get_param("~dog_odom_topic", "/dog_pos"))
        self.vision_topic = str(rospy.get_param("~vision_topic", "/target_ekf_odom"))
        self.uwb_topic = str(rospy.get_param("~uwb_topic", "/AOA_Tag_data"))
        self.output_topic = str(rospy.get_param("~output_odom_topic", "/quaddock/target_odom"))
        self.legacy_output_topic = str(rospy.get_param("~legacy_output_topic", "/quaddock/legacy_preview"))
        self.status_topic = str(rospy.get_param("~status_topic", "/quaddock/ours_sw3/status"))
        self.frame_id = str(rospy.get_param("~output_frame", "world"))
        self.child_frame_id = str(rospy.get_param("~output_child_frame", "target"))
        self.initial_yaw = float(rospy.get_param("~initial_yaw", 0.0))
        self.initial_b = list(rospy.get_param("~initial_translation", [0.0, 0.0, 0.0]))
        self.wait_for_seed = bool(rospy.get_param("~wait_for_seed", False))
        self.external_bearing_gate = bool(rospy.get_param("~external_bearing_gate", False))
        self.bearing_gate_timeout = float(rospy.get_param("~bearing_gate_timeout_s", 0.25))
        self.max_publish_odom_age = float(rospy.get_param("~max_publish_odom_age_s", 0.5))
        self.seed = None
        self.seed_epoch = None
        self.seed_stamp = None
        self.bearing_enabled = False
        self.bearing_gate_arrival = None
        self.histories = {"uav": OdomHistory(), "dog": OdomHistory()}
        self.latest_events = {}
        self.estimator = None
        self.sequence = 0
        self.counters = dict(uav_odom=0, dog_odom=0, vision=0, uwb=0,
                             uwb_range=0, uwb_bearing=0, fused_vision=0,
                             fused_uwb_range=0, fused_uwb_bearing=0,
                             rejected=0, unmatched=0, duplicate=0,
                             bearing_gate_rejected=0)
        self.last_error = ""
        self.last_snapshot = None

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        self.legacy_pub = rospy.Publisher(self.legacy_output_topic, Odometry, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=2, latch=True)
        self.alignment_pub = rospy.Publisher(str(rospy.get_param(
            "~alignment_topic", "/dog_pos_processor/sw3/alignment")), Odometry, queue_size=1)
        if self.wait_for_seed:
            rospy.Subscriber(str(rospy.get_param("~seed_topic", "/dog_pos_processor/sw3/seed")),
                             Odometry, self._seed_cb, queue_size=1)
        if self.external_bearing_gate:
            rospy.Subscriber(str(rospy.get_param("~bearing_gate_topic", "/dog_pos_processor/sw3/bearing_enabled")),
                             Bool, self._bearing_gate_cb, queue_size=1)
        rospy.Subscriber(self.uav_topic, Odometry, self._uav_cb, queue_size=50, tcp_nodelay=True)
        rospy.Subscriber(self.dog_topic, Odometry, self._dog_cb, queue_size=50, tcp_nodelay=True)
        rospy.Subscriber(self.vision_topic, Odometry, self._vision_cb, queue_size=30, tcp_nodelay=True)
        rospy.Subscriber(self.uwb_topic, Odometry, self._uwb_cb, queue_size=30, tcp_nodelay=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.output_hz, 1.0)), self._publish_timer)
        rospy.on_shutdown(self._shutdown)
        self._publish_status("waiting_for_seed" if self.wait_for_seed else "waiting_for_odom")

    def _bearing_gate_cb(self, msg):
        with self.lock:
            self.bearing_enabled = bool(msg.data)
            self.bearing_gate_arrival = rospy.Time.now().to_sec()

    def _seed_cb(self, msg):
        try:
            epoch = float(msg.twist.twist.angular.x)
            if not math.isfinite(epoch) or epoch < 1 or epoch != int(epoch):
                raise ValueError("seed epoch must be a positive integer")
            _finite_vector([msg.pose.pose.position.x, msg.pose.pose.position.y,
                            msg.pose.pose.position.z], 3)
            if _quat_xyzw(msg) is None:
                raise ValueError("seed quaternion must be nonzero")
            with self.lock:
                if self.seed_epoch == int(epoch):
                    return
                self.seed = copy.deepcopy(msg)
                self.seed_epoch = int(epoch)
                self.seed_stamp = _stamp(msg)
                self.estimator = None
                self.last_snapshot = None
                self.bearing_enabled = False
                self.received_relative.clear()
                self.pending = [entry for entry in self.pending
                                if _stamp(entry[1]) >= self.seed_stamp]
                self._try_seed()
        except Exception as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            rospy.logerr("Ours-SW3 seed rejected: %s", self.last_error)

    def _try_seed(self):
        if not self.wait_for_seed or self.seed is None or self.estimator is not None:
            return
        if any(self.histories[key].latest() is None for key in ("dog", "uav")):
            return
        dog = self.histories["dog"].latest()
        uav = self.histories["uav"].latest()
        psi = _yaw_from_odom(self.seed, "quaternion")
        p = self.seed.pose.pose.position
        # Convert the original rigid transform to the core's reference convention.
        b = np.array([p.x, p.y, p.z]) + rz(psi).dot(dog["position"]) - np.asarray(uav["position"])
        initialization = dict(reference_time=self.seed_stamp, psi=psi, b=b.tolist(),
                              rate=[0.0] * 4, q_reference=list(dog["position"]),
                              u_reference=list(uav["position"]))
        self.estimator = make_estimator(self.config, initialization)
        for key in ("dog_odom", "uav_odom"):
            self.estimator.ingest(self.latest_events[key])
        self.last_error = ""
        self._publish_status("initialized_from_original_seed")

    def _event_id(self, sensor, stamp):
        self.sequence += 1
        return "%s:%0.9f:%d" % (sensor, float(stamp), self.sequence)

    def _odom_value(self, msg, sensor):
        stamp = _stamp(msg)
        position = _finite_vector([msg.pose.pose.position.x, msg.pose.pose.position.y,
                                   msg.pose.pose.position.z], 3)
        velocity = _finite_vector([msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                                   msg.twist.twist.linear.z], 3)
        value = dict(position=position.tolist(), velocity=velocity.tolist(),
                     yaw=_yaw_from_odom(msg, self.dog_yaw_mode if sensor == "dog" else "quaternion"))
        if sensor == "uav":
            value["quaternion_xyzw"] = _quat_xyzw(msg)
        return stamp, value

    def _handle_odom(self, sensor, msg):
        try:
            stamp, value = self._odom_value(msg, sensor)
            event = dict(measurement_id=self._event_id(sensor + "_odom", stamp),
                         sensor=sensor + "_odom", sample_time=stamp,
                         arrival_time=rospy.Time.now().to_sec(), value=value)
            with self.lock:
                if not self.histories[sensor].add(stamp, value):
                    self.counters["duplicate"] += 1
                    return
                latest = self.latest_events.get(sensor + "_odom")
                if latest is None or stamp >= latest["sample_time"]:
                    self.latest_events[sensor + "_odom"] = event
                self.counters[sensor + "_odom"] += 1
                if sensor == "uav":
                    self.last_uav_arrival = rospy.Time.now().to_sec()
                    self.uav_received = True
                if self.estimator is not None:
                    response = self.estimator.ingest(event)
                    if not response.get("accepted", True):
                        self.counters["rejected"] += 1
                else:
                    self._try_seed()
                self._drain_pending(rospy.Time.now().to_sec())
        except Exception as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            rospy.logerr_throttle(2.0, "Ours-SW3 odometry rejected: %s", self.last_error)

    def _uav_cb(self, msg):
        self._handle_odom("uav", msg)

    def _dog_cb(self, msg):
        self._handle_odom("dog", msg)

    def _relative_value(self, sensor, msg, uav, dog):
        if sensor == "uwb":
            distance = float(msg.pose.pose.position.x)
            bearing = float(msg.pose.pose.orientation.x)
            return dict(range=distance, bearing=bearing)
        position = _finite_vector([msg.pose.pose.position.x, msg.pose.pose.position.y,
                                   msg.pose.pose.position.z], 3)
        yaw = _yaw_from_odom(msg, self.vision_yaw_mode)
        if self.vision_input_mode == "world_odom":
            relative_world = position - np.asarray(uav["position"], dtype=float)
            position = uav_rotation(uav).T.dot(relative_world)
            yaw = float(wrap(yaw - uav["yaw"]))
        elif self.vision_input_mode != "relative_odom":
            raise ValueError("vision_input_mode must be world_odom or relative_odom")
        return dict(position=position.tolist(), yaw=yaw)

    def _update_availability(self, now):
        if self.aoa_status_tick is None or now - self.aoa_status_tick >= 0.1:
            self.aoa_status_tick = now
            if self.aoa_seq != self.last_aoa_seq:
                self.aoa_received = True
                self.last_aoa_seq = self.aoa_seq
            elif self.last_aoa_arrival is None or now - self.last_aoa_arrival >= 0.5:
                self.aoa_received = False
        if self.uav_status_tick is None or now - self.uav_status_tick >= 0.1:
            self.uav_status_tick = now
            if self.last_uav_arrival is None or now - self.last_uav_arrival >= 0.5:
                self.uav_received = False

    def _bearing_allowed(self, now, uav):
        """Mirror the real processor's readiness and ±45° heading test."""
        if self.external_bearing_gate:
            age = None if self.bearing_gate_arrival is None else now - self.bearing_gate_arrival
            return (self.estimator is not None and self.bearing_enabled and
                    age is not None and 0.0 <= age <= self.bearing_gate_timeout)
        self._update_availability(now)
        if self.estimator is None or not self.aoa_received or not self.uav_received:
            return False
        try:
            snapshot = self.estimator.predict(float(now))
            target = snapshot.get("position")
            if target is None:
                return False
            delta = np.asarray(target, dtype=float) - np.asarray(uav["position"], dtype=float)
            if np.linalg.norm(delta[:2]) < 1e-9:
                return False
            heading = math.atan2(float(delta[1]), float(delta[0]))
            diff = float(wrap(heading - float(uav["yaw"])))
            return abs(diff) <= math.radians(45.0)
        except Exception:
            return False

    def _handle_relative(self, sensor, msg):
        arrival = rospy.Time.now().to_sec()
        with self.lock:
            self.counters[sensor] += 1
            # Same stamped packet can be published twice by linktarck.py.
            # Never turn a replayed packet into independent information.
            if msg.header.stamp.to_sec() > 0.0:
                wire = io.BytesIO()
                # ROS increments Header.seq on each publish, even when linktarck
                # republishes one stamped packet. Sequence is not new evidence.
                canonical = copy.deepcopy(msg)
                canonical.header.seq = 0
                canonical.serialize(wire)
                identity = (sensor, hashlib.sha256(wire.getvalue()).digest())
                if identity in self.received_relative:
                    self.counters["duplicate"] += 1
                    return
                self.received_relative.add(identity)
            self.pending.append((sensor, copy.deepcopy(msg), arrival))
            self._drain_pending(arrival)

    def _drain_pending(self, now):
        waiting = []
        for sensor, msg, arrival in self.pending:
            stamp = _stamp(msg)
            uav = self.histories["uav"].match(stamp, self.max_odom_gap)
            dog = self.histories["dog"].match(stamp, self.max_odom_gap)
            if uav is None or dog is None:
                if now - arrival >= self.match_wait_s:
                    self.counters["unmatched"] += 1
                else:
                    waiting.append((sensor, msg, arrival))
                continue
            self._process_relative(sensor, msg, arrival, now)
        self.pending = waiting

    def _process_relative(self, sensor, msg, arrival, processing_time):
        try:
            stamp = _stamp(msg)
            with self.lock:
                if self.wait_for_seed and (self.estimator is None or stamp < self.seed_stamp):
                    self.counters["rejected"] += 1
                    return
                uav = self.histories["uav"].match(stamp, self.max_odom_gap)
                dog = self.histories["dog"].match(stamp, self.max_odom_gap)
                if uav is None or dog is None:
                    self.counters["unmatched"] += 1
                    return
                value = self._relative_value(sensor, msg, uav, dog)
                if sensor == "uwb":
                    self.aoa_seq += 1
                    self.last_aoa_arrival = arrival
                    self.aoa_received = True
                    if self.estimator is None:
                        return
                    odom = dict(uav=uav, dog=dog)
                    range_event = dict(measurement_id=self._event_id("uwb_range", stamp),
                                       sensor="uwb_range", sample_time=stamp,
                                       arrival_time=arrival, processing_time=processing_time,
                                       value={"range": value["range"]}, odom=odom)
                    if math.isfinite(value["range"]):
                        range_response = self.estimator.ingest(range_event)
                        self.counters["uwb_range"] += 1
                        if range_response.get("accepted", False):
                            self.counters["fused_uwb_range"] += 1
                        else:
                            self.counters["rejected"] += 1
                    else:
                        self.counters["rejected"] += 1
                    # Bearing keeps the original upper-layer heading gate;
                    # range is deliberately independent of this decision.
                    if self._bearing_allowed(processing_time, uav) and math.isfinite(value["bearing"]):
                        bearing_event = dict(measurement_id=self._event_id("uwb_bearing", stamp),
                                             sensor="uwb_bearing", sample_time=stamp,
                                             arrival_time=arrival, processing_time=processing_time,
                                             value={"bearing": value["bearing"]}, odom=odom)
                        bearing_response = self.estimator.ingest(bearing_event)
                        self.counters["uwb_bearing"] += 1
                        if bearing_response.get("accepted", False):
                            self.counters["fused_uwb_bearing"] += 1
                        else:
                            self.counters["rejected"] += 1
                    else:
                        self.counters["bearing_gate_rejected"] += 1
                    return
                event = dict(measurement_id=self._event_id(sensor, stamp), sensor=sensor,
                             sample_time=stamp, arrival_time=arrival,
                             processing_time=processing_time, value=value,
                             odom=dict(uav=uav, dog=dog))
                if self.estimator is None:
                    if sensor != "vision":
                        return
                    self._initialize(event, uav, dog)
                    self.estimator.seen.add(("vision", str(event["measurement_id"])))
                    self.estimator.latest["vision"] = stamp
                    self._publish_status("initialized")
                    return
                response = self.estimator.ingest(event)
                if response.get("accepted", False):
                    self.counters["fused_" + sensor] += 1
                else:
                    self.counters["rejected"] += 1
        except Exception as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            rospy.logerr_throttle(2.0, "Ours-SW3 relative measurement rejected: %s", self.last_error)

    def _vision_cb(self, msg):
        self._handle_relative("vision", msg)

    def _uwb_cb(self, msg):
        self._handle_relative("uwb", msg)

    def _initialize(self, event, uav, dog):
        value = event["value"]
        psi = self.initial_yaw
        if value.get("yaw") is not None:
            psi = float(wrap(value["yaw"] + uav["yaw"] - dog["yaw"]))
        # Invert the same lever-arm observation model used by the solver.
        body_yaw = dog["yaw"] + psi
        b = (uav_rotation(uav).dot(np.asarray(value["position"], dtype=float) +
             np.asarray(self.config.get("vision_uav_lever_arm_m", [0, 0, 0]))) -
             rz(body_yaw).dot(np.asarray(self.config.get("vision_dog_lever_arm_m", [0, 0, 0]))))
        if not np.isfinite(b).all():
            b = np.asarray(self.initial_b, dtype=float)
        initialization = dict(reference_time=event["sample_time"], psi=psi,
                              b=b.tolist(), rate=[0.0, 0.0, 0.0, 0.0],
                              q_reference=list(dog["position"]),
                              u_reference=list(uav["position"]))
        self.estimator = make_estimator(self.config, initialization)
        for key in ("uav_odom", "dog_odom"):
            cached = self.latest_events.get(key)
            if cached is not None:
                self.estimator.ingest(cached)

    @staticmethod
    def _set_pose(msg, position, yaw):
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = [float(v) for v in position]
        q = _yaw_quaternion(yaw)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = q[0], q[1]
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q[2], q[3]

    def _publish_timer(self, _event):
        with self.lock:
            now = rospy.Time.now().to_sec()
            self._update_availability(now)
            self._drain_pending(now)
            if self.estimator is None:
                self._publish_status("waiting_for_seed" if self.wait_for_seed else "waiting_for_vision_initialization")
                return
            now = rospy.Time.now().to_sec()
            if not self._odometry_fresh(now):
                self._publish_status("stale_odometry")
                return
            try:
                snapshot = self.estimator.predict(now)
                self.last_snapshot = snapshot
                self._publish_snapshot(snapshot, now)
                self._publish_status(snapshot.get("status", "unknown"))
            except Exception as exc:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                self._publish_status("estimator_error")
                rospy.logerr_throttle(2.0, "Ours-SW3 prediction failed: %s", self.last_error)

    def _odometry_fresh(self, now):
        for key in ("dog_odom", "uav_odom"):
            latest = self.latest_events.get(key)
            if latest is None:
                return False
            age = float(now) - latest["sample_time"]
            if age < -0.01 or age > self.max_publish_odom_age:
                return False
        return True

    def _publish_alignment(self, snapshot, now):
        if not self.wait_for_seed or self.seed_epoch is None:
            return
        psi = float(snapshot["psi"])
        qref = np.asarray(snapshot["q_reference"], dtype=float)
        translation = (np.asarray(snapshot["u_reference"], dtype=float) +
                       np.asarray(snapshot["b"], dtype=float) - rz(psi).dot(qref))
        rate = np.asarray(snapshot["rate"], dtype=float)
        translation_rate = rate[1:4] - rate[0] * drz(psi).dot(qref)
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(float(now))
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = "dog_odom"
        self._set_pose(msg, translation, psi)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = translation_rate.tolist()
        # Header.seq is rewritten by roscpp: carry the reset epoch explicitly.
        msg.twist.twist.angular.x = float(self.seed_epoch)
        msg.twist.twist.angular.z = float(rate[0])
        self.alignment_pub.publish(msg)

    def _publish_snapshot(self, snapshot, now):
        position = snapshot.get("position")
        velocity = snapshot.get("velocity")
        yaw = snapshot.get("yaw")
        if position is None or velocity is None or yaw is None:
            return
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(float(now))
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        self._set_pose(msg, position, yaw)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = [float(v) for v in velocity]
        rate = snapshot.get("rate") or [0.0, 0.0, 0.0, 0.0]
        # rate[0] is alignment drift, not the dog's total angular velocity.
        covariance = snapshot.get("covariance")
        if covariance is not None:
            cov = np.asarray(covariance, dtype=float)
            dog, _ = self.estimator._odometry_at(now)
            dq = np.asarray(dog["position"]) - np.asarray(snapshot["q_reference"])
            mapping = np.zeros((6, 8))
            mapping[:3, 0] = drz(snapshot["psi"]).dot(dq)
            mapping[:3, 1:4] = np.eye(3)
            mapping[5, 0] = 1.0
            msg.pose.covariance = (mapping.dot(cov).dot(mapping.T)).reshape(-1).tolist()
        self.pub.publish(msg)
        self._publish_alignment(snapshot, now)

        legacy = Odometry()
        legacy.header = msg.header
        legacy.child_frame_id = msg.child_frame_id
        legacy.pose.pose.position.x = msg.pose.pose.position.x
        legacy.pose.pose.position.y = msg.pose.pose.position.y
        legacy.pose.pose.position.z = msg.pose.pose.position.z
        # Preview only. The original processor owns readiness, acceleration,
        # and yaw-rate; estimator status must never manufacture those flags.
        legacy.pose.pose.orientation.w = float(snapshot.get("position_ready", False))
        legacy.pose.pose.orientation.x = float(snapshot.get("yaw_ready", False))
        acceleration = snapshot.get("acceleration_world", [0.0, 0.0, 0.0])
        legacy.pose.pose.orientation.y = float(acceleration[0])
        legacy.pose.pose.orientation.z = float(acceleration[1])
        legacy.twist.twist.linear.x = msg.twist.twist.linear.x
        legacy.twist.twist.linear.y = msg.twist.twist.linear.y
        legacy.twist.twist.linear.z = msg.twist.twist.linear.z
        legacy.twist.twist.angular.x = float(yaw)
        legacy.twist.twist.angular.y = float(snapshot.get("dog_yaw_rate", 0.0))
        self.legacy_pub.publish(legacy)

    def _publish_status(self, status):
        snapshot = self.last_snapshot or {}
        ages = dict(snapshot.get("sensor_ages", {}))
        if self.estimator is not None:
            now = rospy.Time.now().to_sec()
            for sensor in ("vision", "uwb_range", "uwb_bearing", "dog_odom", "uav_odom"):
                stamp = self.estimator.latest.get(sensor)
                ages[sensor] = None if stamp is None else max(0.0, now - stamp)
        payload = dict(status=status, method="ours_sw3_frozen", counters=self.counters,
                       seed_epoch=self.seed_epoch,
                       aoa_received=self.aoa_received,
                       uav_received=self.uav_received,
                       sensor_ages=ages,
                       solve_ms=snapshot.get("solve_ms"),
                       diagnostics=snapshot.get("diagnostics", {}),
                       last_error=self.last_error, timestamp=time.time())
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _shutdown(self):
        try:
            self._publish_status("shutdown")
        except Exception:
            pass


def main():
    rospy.init_node("ours_sw3_estimator", anonymous=False)
    OursSW3Node()
    rospy.spin()


if __name__ == "__main__":
    main()
