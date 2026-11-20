#!/usr/bin/env python3
"""Isolated ROS recording smoke test; --hardware tests the Vicon receiver only."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--stop-signal", choices=["TERM", "INT", "HUP"])
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    output = Path(tempfile.mkdtemp(prefix="quaddock_recording_smoke_"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(ROS_MASTER_URI="http://127.0.0.1:%d" % port,
                      ROS_HOSTNAME="127.0.0.1", ROS_HOME=str(output / "ros_home"),
                      ROS_LOG_DIR=str(output / "ros_logs"))
    os.environ.pop("ROS_IP", None)
    children, logs = [], []

    def start(name, command):
        log = (output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    try:
        start("master", ["roscore", "-p", str(port)])
        import rosgraph
        deadline = time.monotonic() + 10
        while True:
            try:
                rosgraph.Master("/smoke_test").getPid()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("Isolated ROS master failed; logs in " + str(output))
                time.sleep(0.1)
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        rospy.init_node("recording_smoke_test", disable_signals=True)
        publishers = []
        if args.hardware:
            start("receiver", ["roslaunch", "mocap_vicon", "vicon.launch"])
            time.sleep(3)
        else:
            publishers = [rospy.Publisher("/dog_pos", Odometry, queue_size=10),
                          rospy.Publisher("/evaluation/vicon/drone00/pose", PoseStamped, queue_size=10),
                          rospy.Publisher("/evaluation/vicon/other_body/pose", PoseStamped, queue_size=10),
                          rospy.Publisher("/evaluation/vicon/drone00/odom", Odometry, queue_size=10)]
            from sensor_msgs.msg import Image
            image_pub = rospy.Publisher("/camera/infra1/image_rect_raw", Image, queue_size=10)
        recorder_command = ["/usr/bin/python3",
                            str(workspace / "src/quaddock_logging/scripts/record_experiment.py")]
        if not args.hardware:
            # Exercise the actual tracker lifecycle, replacing only its planner
            # launch with an inert process on this isolated ROS master.
            mock = output / "mock_roslaunch.py"
            mock.write_text("import signal,sys,time\n"
                            "assert sys.argv[1:] == ['planning', 'traj_server.launch']\n"
                            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                            "started = time.monotonic()\n"
                            "while time.monotonic() - started < 11: time.sleep(0.1)\n")
            os.environ["BASH_FUNC_roslaunch%%"] = "() { exec /usr/bin/python3 " + str(mock) + ' "$@"; }'
            recorder_command = ["bash", str(workspace / "tracker.sh")]
        if args.with_images:
            recorder_command.append("--with-images")
        recorder = start("recorder", recorder_command + [
                         "--workspace", str(workspace), "--output-dir", str(output / "runs"),
                         "--label", "hardware_receiver_smoke" if args.hardware else "synthetic_smoke",
                         "--scene", "receiver_only_no_motion" if args.hardware else "synthetic_test",
                         "--duration", "7"])
        started = time.monotonic()
        late = None
        signal_sent = False
        while recorder.poll() is None and time.monotonic() - started < 20:
            if publishers:
                odom = Odometry()
                # Constant stamp tests duplicate-stamp reporting and stamp preservation.
                odom.header.stamp = rospy.Time.from_sec(123.0)
                odom.header.frame_id = "test_world"
                odom.pose.pose.orientation.w = 1.0
                pose = PoseStamped(header=odom.header, pose=odom.pose.pose)
                publishers[0].publish(odom)
                publishers[1].publish(pose)
                publishers[2].publish(pose)
                publishers[3].publish(odom)
                image_pub.publish(Image(header=odom.header, height=1, width=1,
                                        encoding="mono8", step=1, data=b"\x00"))
                if time.monotonic() - started > 2 and late is None:
                    late = rospy.Publisher("/evaluation/vicon/quadruped00/pose", PoseStamped,
                                           queue_size=10, latch=True)
                    late.publish(pose)
                    start("event", ["/usr/bin/python3",
                          str(workspace / "src/quaddock_logging/scripts/mark_event.py"),
                          "vision_mask_start", "annotation_only_test"])
            if args.stop_signal and not signal_sent and time.monotonic() - started > 5.5:
                recorder.send_signal(getattr(signal, "SIG" + args.stop_signal))
                signal_sent = True
            time.sleep(0.05)
        expected_exit = 128 + getattr(signal, "SIG" + args.stop_signal) if signal_sent and not args.hardware else 0
        assert recorder.wait(timeout=5) == expected_exit, "Recorder failed; inspect " + str(output)
        run_dir, = (output / "runs").iterdir()
        metadata = json.loads((run_dir / "run.json").read_text())
        assert metadata["state"] == "closed", metadata
        assert not metadata["active_files_remaining"], metadata
        import rosbag
        messages = []
        for name in metadata["bag_files"]:
            with rosbag.Bag(str(run_dir / name)) as bag:
                messages.extend(bag.read_messages())
        topics = {item.topic for item in messages}
        for topic in ["metadata", "health", "events"]:
            assert "/quaddock/experiment/" + topic in topics, topics
        assert not any(item.topic in {"/position_cmd", "/px4ctrl/takeoff_land"} for item in messages)
        if not args.hardware:
            assert "/evaluation/vicon/quadruped00/pose" in topics, topics
            assert metadata["images_recorded"] == args.with_images
            assert ("/camera/infra1/image_rect_raw" in topics) == args.with_images
            assert not topics.intersection({"/evaluation/vicon/other_body/pose",
                                           "/evaluation/vicon/drone00/odom"}), topics
            assert metadata["final_health"]["/dog_pos"]["duplicate_stamps"] > 0
            assert all(item.message.header.stamp.to_sec() == 123.0
                       for item in messages if item.topic == "/dog_pos")
            assert any("vision_mask_start" in item.message.data for item in messages
                       if item.topic == "/quaddock/experiment/events")
        counts = {topic: sum(item.topic == topic for item in messages) for topic in sorted(topics)}
        summary = dict(hardware=args.hardware, output=str(output), counts=counts)
        if args.hardware:
            diagnostic = [item.message for item in messages if item.topic == "/evaluation/vicon/diagnostics"]
            assert diagnostic, "No Vicon diagnostics; inspect receiver.log in " + str(output)
            assert "/evaluation/vicon/drone00/pose" in topics, (
                "No drone00 positions recorded; inspect receiver.log in " + str(output))
            summary["last_vicon_status"] = [
                dict(name=s.name, message=s.message, values={kv.key: kv.value for kv in s.values})
                for s in diagnostic[-1].status]
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        for process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
