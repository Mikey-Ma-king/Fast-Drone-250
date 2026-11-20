#!/usr/bin/env python3
"""End-to-end check on a private ROS master, using synthetic messages only.

Run after building and sourcing devel/setup.bash. This script starts only the
two nodes in dog_pos_processor_new.launch; it never starts hardware/control.
"""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest


class RosBridgeCheck(unittest.TestCase):
    def test_bridge_lifecycle_and_output_contract(self):
        # Never connect this test's publishers to the user's current ROS master.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="sw3_ros_check_") as directory:
            previous_env = dict(os.environ)
            os.environ.update(ROS_MASTER_URI="http://127.0.0.1:%d" % port,
                              ROS_IP="127.0.0.1", ROS_HOME=directory,
                              ROS_LOG_DIR=directory + "/logs",
                              OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            os.environ.pop("ROS_HOSTNAME", None)
            processes = []
            log_path = Path(directory) / "process.log"
            try:
                with log_path.open("w") as log:
                    processes.append(subprocess.Popen(["roscore", "-p", str(port)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    import rosgraph
                    deadline = time.monotonic() + 15
                    master = rosgraph.Master("/sw3_bridge_check")
                    while True:
                        try:
                            master.getPid()
                            break
                        except Exception:
                            if time.monotonic() > deadline:
                                self.fail("private ROS master did not start")
                            time.sleep(0.05)
                    processes.append(subprocess.Popen(
                        ["roslaunch", "planning", "dog_pos_processor_new.launch"],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    self._exercise(master)
            except Exception:
                print(log_path.read_text()[-6000:])
                raise
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=3)
                os.environ.clear()
                os.environ.update(previous_env)

    def _exercise(self, master):
        import rospy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String, Float64
        from quadrotor_msgs.msg import TakeoffLand

        rospy.init_node("sw3_bridge_check", anonymous=True, disable_signals=True)
        self.addCleanup(rospy.signal_shutdown, "private check complete")
        messages, statuses, seeds = [], [], []
        lock = threading.Lock()

        def capture(destination, value):
            with lock:
                destination.append(value)

        subscribers = [
            rospy.Subscriber("/dog_pos_processed", Odometry, lambda m: capture(messages, m)),
            rospy.Subscriber("/quaddock/ours_sw3/status", String,
                             lambda m: capture(statuses, json.loads(m.data))),
            rospy.Subscriber("/dog_pos_processor/sw3/seed", Odometry, lambda m: capture(seeds, m)),
        ]
        dog = rospy.Publisher("/dog_pos", Odometry, queue_size=20)
        uav = rospy.Publisher("/vins_fusion/imu_propagate", Odometry, queue_size=20)
        vision = rospy.Publisher("/target_ekf_odom", Odometry, queue_size=20)
        aoa = rospy.Publisher("/AOA_Tag_data", Odometry, queue_size=20)
        takeoff = rospy.Publisher("/px4ctrl/takeoff_land", TakeoffLand, queue_size=1)
        saved_yaw = rospy.Publisher("/yaw_diff_preset", Float64, queue_size=1)

        deadline = time.monotonic() + 15
        while not all(p.get_num_connections() for p in (dog, uav, vision, aoa, takeoff, saved_yaw)):
            self.assertLess(time.monotonic(), deadline, "bridge subscribers did not connect")
            time.sleep(0.03)

        def drive(seconds, visual=False, ranging=False, uav_yaw=0.0,
                  dog_yaw=0.0, body_speed=0.0, send_uav=True):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                stamp = rospy.Time.now()
                um = Odometry()
                um.header.stamp = stamp
                um.pose.pose.position.z = 1.0
                um.pose.pose.orientation.z = math.sin(uav_yaw / 2)
                um.pose.pose.orientation.w = math.cos(uav_yaw / 2)
                if send_uav:
                    uav.publish(um)
                dm = Odometry()
                dm.header.stamp = stamp
                dm.pose.pose.position.x, dm.pose.pose.position.z = 2.0, 0.42
                dm.pose.pose.orientation.w = dog_yaw
                dm.twist.twist.linear.x = body_speed
                dog.publish(dm)
                if visual:
                    vm = Odometry()
                    # The old frontend can stamp before detection finishes.
                    # C++ compensates it and dates the resulting state at receipt.
                    vm.header.stamp = stamp - rospy.Duration(0.2)
                    vm.pose.pose.position.x, vm.pose.pose.position.z = 1.5, 0.42
                    vm.pose.pose.orientation.w = dog_yaw
                    vision.publish(vm)
                if ranging:
                    am = Odometry()
                    am.header.stamp = stamp
                    am.pose.pose.position.x = math.hypot(1.5, 0.58)
                    am.pose.pose.orientation.x = -uav_yaw
                    aoa.publish(am)
                    aoa.publish(am)  # linktarck can repeat the same stamped packet.
                time.sleep(0.05)

        # Preserve the original startup before any visual detection is available.
        drive(1.0)
        self.assertTrue(messages and statuses and seeds)
        self.assertEqual(messages[-1].pose.pose.orientation.w, 1.0)
        self.assertEqual(messages[-1].pose.pose.orientation.x, 1.0)
        self.assertAlmostEqual(messages[-1].pose.pose.position.x, 0.0, delta=0.02)
        first_epoch = statuses[-1]["seed_epoch"]

        # Vision corrects the seed; ranges closer than the old 3 m cutoff fuse.
        drive(3.0, visual=True, ranging=True)
        self.assertAlmostEqual(messages[-1].pose.pose.position.x, 1.5, delta=0.10)
        self.assertAlmostEqual(messages[-1].pose.pose.position.z, 0.42, delta=0.10)
        counters = statuses[-1]["counters"]
        self.assertGreater(counters["fused_vision"], 10)
        self.assertGreater(counters["fused_uwb_range"], 10)
        self.assertGreater(counters["fused_uwb_bearing"], 10)
        self.assertGreater(counters["duplicate"], 10)

        # Turning away must reject bearing without discarding range.
        drive(0.5, visual=True, ranging=True, uav_yaw=math.pi)
        before = dict(statuses[-1]["counters"])
        drive(0.8, visual=True, ranging=True, uav_yaw=math.pi)
        after = statuses[-1]["counters"]
        self.assertGreater(after["fused_uwb_range"], before["fused_uwb_range"] + 5)
        self.assertEqual(after["fused_uwb_bearing"], before["fused_uwb_bearing"])
        self.assertGreater(after["bearing_gate_rejected"], before["bearing_gate_rejected"])

        # The saved yaw difference and takeoff reset must reseed without vision.
        saved_yaw.publish(Float64(data=0.3))
        time.sleep(0.1)
        takeoff.publish(TakeoffLand(takeoff_land_cmd=1))
        drive(1.0)
        self.assertGreater(statuses[-1]["seed_epoch"], first_epoch)
        self.assertAlmostEqual(messages[-1].pose.pose.position.x, 0.0, delta=0.03)
        self.assertAlmostEqual(messages[-1].twist.twist.angular.x, 0.3, delta=0.03)

        # Replay an old epoch: it must never move the new output to the old transform.
        replay = rospy.Publisher("/dog_pos_processor/sw3/alignment", Odometry, queue_size=1)
        time.sleep(0.15)
        old = Odometry()
        old.header.stamp = rospy.Time.now()
        old.pose.pose.position.x = 50
        old.pose.pose.orientation.w = 1
        old.twist.twist.angular.x = first_epoch
        replay.publish(old)
        count = len(messages)
        drive(0.3)
        self.assertTrue(all(abs(m.pose.pose.position.x) < 1 for m in messages[count:]))

        # Missing UAV odometry stops fresh alignment and clears output-ready fields.
        drive(1.0, send_uav=False)
        self.assertEqual(statuses[-1]["status"], "stale_odometry")
        self.assertEqual(messages[-1].pose.pose.orientation.w, 0.0)
        self.assertEqual(messages[-1].pose.pose.orientation.x, 0.0)
        drive(0.8)
        self.assertEqual(messages[-1].pose.pose.orientation.w, 1.0)

        # Body-frame velocity must keep the original world-frame output convention.
        drive(0.6, uav_yaw=math.pi/2, dog_yaw=math.pi/2)
        saved_yaw.publish(Float64(data=0.0))
        time.sleep(0.1)
        takeoff.publish(TakeoffLand(takeoff_land_cmd=1))
        drive(0.8, uav_yaw=math.pi/2, dog_yaw=math.pi/2, body_speed=0.2)
        self.assertAlmostEqual(messages[-1].twist.twist.linear.x, 0.0, delta=0.02)
        self.assertAlmostEqual(messages[-1].twist.twist.linear.y, 0.2, delta=0.02)
        self.assertEqual(statuses[-1]["last_error"], "")

        publishers = dict(master.getSystemState()[0])
        self.assertEqual(publishers["/dog_pos_processed"], ["/dog_pos_processor"])
        for topic in ("/position_cmd", "/mavros/setpoint_raw/attitude"):
            self.assertNotIn(topic, publishers)
        for m in messages:
            self.assertIn(m.pose.pose.orientation.w, (0.0, 1.0))
            self.assertIn(m.pose.pose.orientation.x, (0.0, 1.0))
            self.assertTrue(all(math.isfinite(v) for v in (
                m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
                m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z,
                m.twist.twist.angular.x, m.twist.twist.angular.y,
                m.pose.pose.orientation.y, m.pose.pose.orientation.z)))
        print("Private ROS bridge checks passed; counters:", statuses[-1]["counters"])
        # Keep subscribers alive through the complete check.
        self.assertEqual(len(subscribers), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
