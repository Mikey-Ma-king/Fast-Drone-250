#!/usr/bin/env python3
import subprocess
import time
import rospy
from std_msgs.msg import String
from quadrotor_msgs.msg import TakeoffLand

# 启动指令

SESSION = "real_fly"

# 启动指令映射
INIT_CMDS = {
    0: "roscore",
    1: "./zydauto0.sh",
    2: "./zydauto1.sh"
}
TAKEOFF_CMD = {
    3: "./zydauto2.sh"
}
WITHDRAW_CMD = {
    5: "./withdraw.sh"
}

# 用于记录启动的进程标识关键字（可用于精准 pkill）
PROC_KEYWORDS = [
    "roscore",
    "zydauto0",
    "zydauto1",
    "zydauto2"
]
kill_all_keywords =[
    "traj_server",
    "nodelet",
    "hc_14",
    "mavros_node",
    "px4ctrl_node",
    "flow_publisher_node",
    "px4",
    "gzserver",
    "svo_node",
    "planning"
]

def tmux_send(cmd, pane_index):
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION}:0.{pane_index}", "C-c"])
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION}:0.{pane_index}", cmd, "C-m"])

def run_init():
    print("[INIT] Starting core nodes...")
    for i, cmd in INIT_CMDS.items():
        tmux_send(cmd, i)
        time.sleep(0.5)

def run_takeoff():
    print("[TAKEOFF] Starting takeoff node...")
    for i, cmd in TAKEOFF_CMD.items():
        tmux_send(cmd, i)

def stop_all():
    print("[LAND] Terminating all relevant processes...")
    for i, cmd in INIT_CMDS.items():
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION}:0.{i}", "C-c"])
        time.sleep(0.1)
    for kw in PROC_KEYWORDS:
        subprocess.run(["pkill", "-f", kw])
    for kw in kill_all_keywords:
        subprocess.run(["killall",kw])
    time.sleep(0.5)
    subprocess.run(["pkill","-9","-f","python3 MPC.py"])
    for i in range(8):
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION}:0.{i}", "C-c"])
        time.sleep(0.1)
    print("[LAND] Done.")

def ros_listener():
    def cb(msg):
        if msg.takeoff_land_cmd == 2:
            stop_all()

    rospy.init_node("real_fly_control_node", anonymous=True)
    rospy.Subscriber("/px4ctrl/takeoff_land", TakeoffLand, cb)
    rospy.spin()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["init", "takeoff", "land", "listen","0","1","2","3"], help="Command to control tmux processes")
    args = parser.parse_args()

    if args.cmd == "init":
        run_init()
    elif args.cmd == "takeoff":
        run_takeoff()
    elif args.cmd == "land":
        stop_all()
    elif args.cmd == "listen":
        ros_listener()

    if args.cmd == "0":
        run_init()
    elif args.cmd == "1":
        run_takeoff()
    elif args.cmd == "2":
        stop_all()
    elif args.cmd == "3":
        ros_listener()

if __name__ == "__main__":
    main()
