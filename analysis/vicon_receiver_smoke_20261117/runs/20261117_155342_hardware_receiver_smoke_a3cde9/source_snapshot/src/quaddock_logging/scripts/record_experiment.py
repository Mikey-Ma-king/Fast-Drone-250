#!/usr/bin/env python3
"""Record experiment inputs/outputs, health and provenance; no control publishers."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
import uuid

import rosgraph
import rospkg
import rospy
from std_msgs.msg import String
import yaml


BASE_TOPIC = "/quaddock/experiment/"
HEADER_TYPES = {"nav_msgs/Odometry", "geometry_msgs/PoseStamped",
                "geometry_msgs/PoseWithCovarianceStamped", "sensor_msgs/Imu",
                "diagnostic_msgs/DiagnosticArray", "svo_msgs/Info"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def git_state(path):
    def run(*args):
        result = subprocess.run(["git", "-C", str(path)] + list(args),
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(head=run("rev-parse", "HEAD"), status=run("status", "--short"))


class Health:
    def __init__(self, topics, patterns):
        self.lock = threading.Lock()
        self.data = {name: self.empty() for name in topics}
        self.subscribers = {}
        self.patterns = patterns
        self.previous_counts = {}
        self.previous_report = time.monotonic()

    @staticmethod
    def empty():
        return dict(messages=0, last_receipt_monotonic=None, last_header_stamp=None,
                    zero_stamps=0, duplicate_stamps=0, backwards_stamps=0,
                    max_receipt_gap_s=0.0, last_header_age_at_receipt_s=None)

    def callback(self, msg, topic):
        now = time.monotonic()
        datatype = msg._connection_header.get("type", "")
        # Only read the fixed Header prefix of known messages. No large image decoding.
        stamp = None
        if datatype in HEADER_TYPES and len(msg._buff) >= 12:
            seconds, nanos = struct.unpack_from("<II", msg._buff, 4)
            stamp = seconds + nanos * 1e-9
        with self.lock:
            data = self.data[topic]
            previous_receipt = data["last_receipt_monotonic"]
            if previous_receipt is not None:
                data["max_receipt_gap_s"] = max(data["max_receipt_gap_s"], now - previous_receipt)
            data["last_receipt_monotonic"] = now
            data["messages"] += 1
            data["type"] = datatype
            if stamp is not None:
                previous = data["last_header_stamp"]
                data["zero_stamps"] += int(stamp == 0)
                if previous is not None:
                    data["duplicate_stamps"] += int(stamp == previous)
                    data["backwards_stamps"] += int(stamp < previous)
                data["last_header_stamp"] = stamp
                data["last_header_age_at_receipt_s"] = rospy.Time.now().to_sec() - stamp

    def discover(self):
        published = rospy.get_published_topics()
        for name, _ in published:
            if any(re.fullmatch(pattern, name) for pattern in self.patterns):
                with self.lock:
                    self.data.setdefault(name, self.empty())
        for name in list(self.data):
            if name not in self.subscribers:
                self.subscribers[name] = rospy.Subscriber(name, rospy.AnyMsg,
                    self.callback, callback_args=name, queue_size=100, tcp_nodelay=True)

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            result = {}
            interval = max(now - self.previous_report, 1e-9)
            for name, data in self.data.items():
                row = dict(data)
                receipt = row.pop("last_receipt_monotonic")
                row["receipt_age_s"] = None if receipt is None else now - receipt
                row["interval_hz"] = (data["messages"] - self.previous_counts.get(name, 0)) / interval
                self.previous_counts[name] = data["messages"]
                result[name] = row
            self.previous_report = now
            return result


def snapshot_context(run_dir, phase):
    prefixes = ["/dog_pos_processor", "/ours_sw3_estimator", "/evaluation/vicon",
                "/vicon", "/svo", "/camera", "/read", "/use_sim_time"]
    params = {name: rospy.get_param(name) for name in prefixes if rospy.has_param(name)}
    (run_dir / ("parameters_" + phase + ".yaml")).write_text(yaml.safe_dump(params, allow_unicode=True))
    master = rosgraph.Master(rospy.get_name())
    graph = dict(published_topics=master.getPublishedTopics("/"), system_state=master.getSystemState())
    write_json(run_dir / ("ros_graph_" + phase + ".json"), graph)


def source_snapshot(run_dir, workspace, config):
    paths = ["zydauto1.sh", "tracker.sh", "schurvins.sh",
             "src/planning/src/dog_pos_processor_sw3.cpp",
             "src/planning/include/dog_pos_processor_sw3.h",
             "src/planning/launch/dog_pos_processor_new.launch",
             "src/quaddock_ours/config/ours_real.yaml",
             "src/quaddock_ours/scripts/ours_sw3_node.py",
             "src/quaddock_ours/src/quaddock_ours/estimator.py",
             "src/quaddock_ours/src/quaddock_ours/models.py",
             "src/read/src/read.cpp",
             "src/motion_capture_system/mocap_vicon/launch/vicon.launch",
             "src/motion_capture_system/mocap_vicon/src/ViconDriver.cpp",
             "src/quaddock_logging/scripts/record_experiment.py"]
    hashes = {}
    for name in paths:
        source = workspace / name
        if source.is_file():
            target = run_dir / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copy2(config, run_dir / "recording_config.yaml")
    return hashes


def main():
    default_config = Path(__file__).resolve().parents[1] / "config/recording.yaml"
    if not default_config.exists():
        default_config = Path(rospkg.RosPack().get_path("quaddock_logging")) / "config/recording.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "perching_bag")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--label", default="vicon")
    parser.add_argument("--scene", default="unconfirmed")
    parser.add_argument("--method", default="ours_sw3", help="Operator label, not proof of an active method")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--duration", type=float, help="Optional wall-clock recording duration, seconds")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    config = yaml.safe_load(args.config.read_text())
    topics = list(config["topics"])
    if not args.no_images:
        topics += config["image_topics"]
    expression = "^(?:" + "|".join([re.escape(name) for name in topics] + config["topic_regex"]) + ")$"
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", args.label)
    run_dir = args.output_dir.expanduser() / (stamp + "_" + label + "_" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    rospy.init_node("quaddock_experiment_recorder", anonymous=True, disable_signals=True)
    metadata_pub = rospy.Publisher(BASE_TOPIC + "metadata", String, queue_size=2, latch=True)
    health_pub = rospy.Publisher(BASE_TOPIC + "health", String, queue_size=2, latch=True)
    event_pub = rospy.Publisher(BASE_TOPIC + "events", String, queue_size=10)
    health = Health(config["health_topics"], [r"/evaluation/vicon/.*", r"/vicon/.*"])
    command = ["rosbag", "record", "--regex", "--tcpnodelay", "--buffsize=256",
               "--chunksize=256", "--split", "--size=2048", "--duration=300",
               "--repeat-latched", "--min-space=1G", "-O", str(run_dir / "data.bag"),
               expression, "__name:=quaddock_bag_" + uuid.uuid4().hex[:8]]
    metadata = dict(schema_version=1, run_directory=str(run_dir), state="starting",
                    start_wall_time=time.time(), start_ros_time=rospy.Time.now().to_sec(),
                    label=args.label, scene=args.scene, requested_method=args.method,
                    mode="online_recording", images_recorded=not args.no_images,
                    evaluation=config["evaluation"], ros_master_uri=os.environ.get("ROS_MASTER_URI"),
                    workspace_git=git_state(args.workspace),
                    source_snapshot_is_running_binary_proof=False,
                    source_sha256=source_snapshot(run_dir, args.workspace, args.config),
                    rosbag_command=command,
                    health_scope="subscriber observations, not proof of messages committed to bag")
    write_json(run_dir / "run.json", metadata)
    snapshot_context(run_dir, "start")
    metadata_pub.publish(String(data=json.dumps(metadata, ensure_ascii=False)))
    print("Recording to " + str(run_dir), flush=True)
    process = None
    exit_code = 0
    try:
        with (run_dir / "rosbag_record.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            metadata["state"] = "recording"
            started = time.monotonic()
            start_event_sent = False
            with (run_dir / "health.jsonl").open("w") as health_log:
                while not stopped.is_set() and not rospy.is_shutdown():
                    if process.poll() is not None:
                        raise RuntimeError("rosbag exited unexpectedly: %s; see rosbag_record.log" % process.returncode)
                    health.discover()
                    payload = dict(ros_time=rospy.Time.now().to_sec(), wall_time=time.time(), streams=health.snapshot())
                    encoded = json.dumps(payload, ensure_ascii=False)
                    health_log.write(encoded + "\n")
                    health_log.flush()
                    health_pub.publish(String(data=encoded))
                    metadata_pub.publish(String(data=json.dumps(metadata, ensure_ascii=False)))
                    if not start_event_sent and event_pub.get_num_connections():
                        event_pub.publish(String(data=json.dumps(dict(kind="record_start", run=str(run_dir),
                                          ros_time=rospy.Time.now().to_sec(), wall_time=time.time()))))
                        start_event_sent = True
                    if args.duration is not None and time.monotonic() - started >= args.duration:
                        break
                    stopped.wait(1.0)
            metadata["state"] = "stopping"
    except Exception as exc:
        metadata["error"] = str(exc)
        metadata["state"] = "failed"
        exit_code = 1
        rospy.logerr(str(exc))
    finally:
        metadata["end_wall_time"] = time.time()
        metadata["end_ros_time"] = rospy.Time.now().to_sec()
        metadata["final_health"] = health.snapshot()
        try:
            snapshot_context(run_dir, "end")
        except Exception as exc:
            metadata["final_context_error"] = str(exc)
        # Keep ROS alive while the recorder drains and closes its bag index.
        event_pub.publish(String(data=json.dumps(dict(kind="record_stop", ros_time=rospy.Time.now().to_sec(),
                                                      wall_time=time.time()))))
        metadata_pub.publish(String(data=json.dumps(metadata, ensure_ascii=False)))
        time.sleep(0.3)
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                metadata["forced_shutdown"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                exit_code = 1
            metadata["rosbag_exit_code"] = process.returncode
            if process.returncode != 0:
                exit_code = 1
        metadata["bag_files"] = [p.name for p in sorted(run_dir.glob("*.bag"))]
        metadata["active_files_remaining"] = [p.name for p in sorted(run_dir.glob("*.active"))]
        if metadata["active_files_remaining"] or not metadata["bag_files"]:
            exit_code = 1
        metadata["state"] = "closed" if exit_code == 0 else "failed"
        write_json(run_dir / "run.json", metadata)
        print("Recorder %s: %s" % (metadata["state"], run_dir), flush=True)
        rospy.signal_shutdown("recording finished")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
