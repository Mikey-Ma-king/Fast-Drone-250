#!/usr/bin/env python3
"""
实验脚本：重复 x 次降落实验，统计降落误差，只保存 JSON（不画图）
流程与 experiment_landing 一致：
1. 运行 fake_target
2. 启动 rosbag record
3. 等待随机时间
4. 发布返航触发
5. 监控降落，记录降落误差（高度=0.4 时的 UAV 与 target 偏差）
6. 停止 rosbag、fake_target，发布停止触发，复位到 (0,0,0.5)，等待稳定
7. 重复共 x 次，保存 results.json（含每次的 error_x, error_y, error_xy, landing_time, success 及汇总统计）
支持 Ctrl+C 安全退出并保存已得结果。
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
FAKE_TARGET_SCRIPT = os.path.join(WORKSPACE, 'fake_target.py')

active_processes = {'fake_target': None, 'rosbag': None}
active_processes_lock = threading.Lock()
shutdown_flag = threading.Event()


class UAVPositionListener:
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]

    def get_position(self):
        with self.position_lock:
            return self.position.copy() if self.position is not None else None


class TargetPositionListener:
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]

    def get_position(self):
        with self.position_lock:
            return self.position.copy() if self.position is not None else None


def publish_trigger(pub, w_value):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = 0.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 0.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = w_value
    pub.publish(msg)
    rospy.loginfo("Published trigger w=%.1f", w_value)


def publish_position_cmd(pub, x, y, z, traj_id=1):
    cmd = PositionCommand()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "world"
    cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
    cmd.trajectory_id = traj_id
    cmd.position.x = x
    cmd.position.y = y
    cmd.position.z = z
    cmd.velocity.x = 0.0
    cmd.velocity.y = 0.0
    cmd.velocity.z = 0.0
    cmd.acceleration.x = 0.0
    cmd.acceleration.y = 0.0
    cmd.acceleration.z = 0.0
    cmd.yaw = 0.0
    cmd.yaw_dot = 0.0
    pub.publish(cmd)
    rospy.loginfo("Published position command (%.2f, %.2f, %.2f)", x, y, z)


def wait_for_landing(uav_listener, target_listener, timeout=60.0):
    """
    等待无人机降落（高度约 0.4 m），或停止移动超过 5 秒。
    返回: (landing_time, uav_pos, target_pos, error_x, error_y, error_xy, stopped)
    """
    start_time = time.time()
    rate = rospy.Rate(50)
    last_position = None
    last_position_time = None
    position_threshold = 0.05
    stop_duration = 5.0

    while time.time() - start_time < timeout:
        if shutdown_flag.is_set():
            uav_pos = uav_listener.get_position()
            target_pos = target_listener.get_position()
            if uav_pos is not None and target_pos is not None:
                ex = uav_pos[0] - target_pos[0]
                ey = uav_pos[1] - target_pos[1]
                return time.time() - start_time, uav_pos, target_pos, ex, ey, math.sqrt(ex**2 + ey**2), False
            return time.time() - start_time, None, None, None, None, None, False

        uav_pos = uav_listener.get_position()
        target_pos = target_listener.get_position()

        if uav_pos is not None:
            current_time = time.time()
            if last_position is not None:
                change = math.sqrt((uav_pos[0] - last_position[0]) ** 2 + (uav_pos[1] - last_position[1]) ** 2)
                if change < position_threshold:
                    if last_position_time is None:
                        last_position_time = current_time
                    elif (current_time - last_position_time) >= stop_duration:
                        rospy.logwarn("UAV stopped moving >%.1fs", stop_duration)
                        if target_pos is not None:
                            ex = uav_pos[0] - target_pos[0]
                            ey = uav_pos[1] - target_pos[1]
                            return time.time() - start_time, uav_pos, target_pos, ex, ey, math.sqrt(ex**2 + ey**2), True
                        return time.time() - start_time, uav_pos, None, None, None, None, True
                else:
                    last_position_time = None
            last_position = uav_pos.copy()

            if target_pos is not None and abs(uav_pos[2] - 0.4) < 0.05:
                landing_time = time.time() - start_time
                ex = uav_pos[0] - target_pos[0]
                ey = uav_pos[1] - target_pos[1]
                e_xy = math.sqrt(ex**2 + ey**2)
                rospy.loginfo("Landing: height=%.3f, error_xy=%.3f m, time=%.2f s", uav_pos[2], e_xy, landing_time)
                return landing_time, uav_pos, target_pos, ex, ey, e_xy, False

        rate.sleep()

    rospy.logwarn("Landing timeout %.1f s", timeout)
    uav_pos = uav_listener.get_position()
    target_pos = target_listener.get_position()
    if uav_pos is not None and target_pos is not None:
        ex = uav_pos[0] - target_pos[0]
        ey = uav_pos[1] - target_pos[1]
        return timeout, uav_pos, target_pos, ex, ey, math.sqrt(ex**2 + ey**2), False
    return timeout, None, None, None, None, None, False


def cleanup_processes():
    with active_processes_lock:
        if active_processes['rosbag'] is not None:
            try:
                active_processes['rosbag'].send_signal(signal.SIGINT)
                active_processes['rosbag'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['rosbag'].kill()
                active_processes['rosbag'].wait()
            except Exception as e:
                rospy.logwarn("Error stopping rosbag: %s", e)
            active_processes['rosbag'] = None
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['fake_target'].kill()
                active_processes['fake_target'].wait()
            except Exception as e:
                rospy.logwarn("Error stopping fake_target: %s", e)
            active_processes['fake_target'] = None
    rospy.loginfo("Processes cleaned up.")


def signal_handler(signum, frame):
    rospy.logwarn("Ctrl+C received. Shutting down safely...")
    shutdown_flag.set()
    cleanup_processes()


def run_single_experiment(experiment_num, total_num, uav_listener, target_listener, trigger_pub, pos_cmd_pub, base_dir,
                          wait_min=5.0, wait_max=35.0, record_rosbag=True):
    """单次降落实验，返回该次降落误差等结果（与 experiment_landing 流程一致）。"""
    exp_dir = os.path.join(base_dir, f"experiment_{experiment_num:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    rospy.loginfo("=" * 60)
    rospy.loginfo("Experiment %d/%d", experiment_num, total_num)
    rospy.loginfo("=" * 60)

    if shutdown_flag.is_set():
        return None

    # 1. 启动 fake_target
    rospy.loginfo("Step 1: Starting fake_target...")
    seed_value = (experiment_num * 1000000 + int(time.time() * 1000)) % (2**32)
    env = os.environ.copy()
    env['FAKE_TARGET_RANDOM_SEED'] = str(seed_value)
    with active_processes_lock:
        active_processes['fake_target'] = subprocess.Popen(
            [sys.executable, FAKE_TARGET_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=WORKSPACE,
        )
    time.sleep(3)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 2. 启动 rosbag
    if record_rosbag:
        rospy.loginfo("Step 2: Starting rosbag record...")
        bag_path = os.path.join(exp_dir, f"experiment_{experiment_num:03d}.bag")
        topics = [
            '/dog_pos', '/mode_manager', '/vins_fusion/imu_propagate', '/position_cmd',
            '/target_ekf_odom', '/dog_pos_processed', '/ground_truth_traj',
        ]
        with active_processes_lock:
            active_processes['rosbag'] = subprocess.Popen(
                ['rosbag', 'record', '-O', bag_path] + topics,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=WORKSPACE,
            )
        time.sleep(1)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 3. 等待随机时间
    wait_time = random.uniform(wait_min, wait_max)
    rospy.loginfo("Step 3: Waiting %.2f s...", wait_time)
    elapsed = 0.0
    step = 0.5
    while elapsed < wait_time and not shutdown_flag.is_set():
        time.sleep(min(step, wait_time - elapsed))
        elapsed += step
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 4. 发布返航触发
    rospy.loginfo("Step 4: Publish return trigger...")
    publish_trigger(trigger_pub, 0.0)
    time.sleep(0.5)

    # 5. 等待降落并记录误差
    rospy.loginfo("Step 5: Monitoring landing...")
    landing_time, uav_pos, target_pos, error_x, error_y, error_xy, stopped = wait_for_landing(
        uav_listener, target_listener
    )
    success = not stopped and error_xy is not None and error_xy < 0.1

    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 6. 停止 rosbag
    if record_rosbag:
        rospy.loginfo("Step 6: Stopping rosbag...")
        with active_processes_lock:
            if active_processes['rosbag'] is not None:
                try:
                    active_processes['rosbag'].send_signal(signal.SIGINT)
                    active_processes['rosbag'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    active_processes['rosbag'].kill()
                    active_processes['rosbag'].wait()
                active_processes['rosbag'] = None
        time.sleep(1)

    # 7. 停止 fake_target
    rospy.loginfo("Step 7: Stopping fake_target...")
    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['fake_target'].kill()
                active_processes['fake_target'].wait()
            active_processes['fake_target'] = None
    time.sleep(1)

    # 8. 发布停止触发
    rospy.loginfo("Step 8: Publish stop trigger...")
    publish_trigger(trigger_pub, -1.0)
    time.sleep(0.5)

    # 9. 复位到 (0, 0, 0.5)
    rospy.loginfo("Step 9: Reset to (0, 0, 0.5)...")
    current_pos = uav_listener.get_position() or [0.0, 0.0, 0.5]
    for _ in range(20):
        publish_position_cmd(pos_cmd_pub, 0.0, 0.0, 0.5)
        time.sleep(0.1)
    time.sleep(3.0)

    # 10. 稳定
    time.sleep(2.0)

    return {
        'experiment_num': experiment_num,
        'waiting_time': wait_time,
        'landing_time': landing_time,
        'success': success,
        'error_x': error_x,
        'error_y': error_y,
        'error_xy': error_xy,
        'uav_pos': uav_pos,
        'target_pos': target_pos,
        'stopped': stopped,
        'exp_dir': exp_dir,
    }


def save_results(results, base_dir):
    """保存实验结果为 JSON（含统计）。"""
    def to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    experiments = []
    for r in results:
        uav = r.get('uav_pos')
        tgt = r.get('target_pos')
        experiments.append({
            'experiment_num': r['experiment_num'],
            'waiting_time': to_float(r.get('waiting_time')),
            'landing_time': to_float(r.get('landing_time')),
            'success': bool(r.get('success', False)),
            'error_x': to_float(r.get('error_x')),
            'error_y': to_float(r.get('error_y')),
            'error_xy': to_float(r.get('error_xy')),
            'stopped': bool(r.get('stopped', False)),
            'uav_position': {'x': to_float(uav[0]) if uav and len(uav) > 0 else None, 'y': to_float(uav[1]) if uav and len(uav) > 1 else None, 'z': to_float(uav[2]) if uav and len(uav) > 2 else None} if uav else None,
            'target_position': {'x': to_float(tgt[0]) if tgt and len(tgt) > 0 else None, 'y': to_float(tgt[1]) if tgt and len(tgt) > 1 else None, 'z': to_float(tgt[2]) if tgt and len(tgt) > 2 else None} if tgt else None,
        })

    valid = [r for r in results if r.get('error_xy') is not None]
    statistics = None
    if valid:
        ex = [r['error_x'] for r in valid if r.get('error_x') is not None]
        ey = [r['error_y'] for r in valid if r.get('error_y') is not None]
        e_xy = [r['error_xy'] for r in valid]
        lt = [r['landing_time'] for r in valid if r.get('landing_time') is not None]
        wt = [r.get('waiting_time') for r in valid if r.get('waiting_time') is not None]
        statistics = {
            'valid_experiments': len(valid),
            'total_experiments': len(results),
            'success_count': sum(1 for r in results if r.get('success')),
            'error_x': {'mean': float(np.mean(ex)), 'std': float(np.std(ex)), 'min': float(np.min(ex)), 'max': float(np.max(ex))} if ex else None,
            'error_y': {'mean': float(np.mean(ey)), 'std': float(np.std(ey)), 'min': float(np.min(ey)), 'max': float(np.max(ey))} if ey else None,
            'error_xy': {'mean': float(np.mean(e_xy)), 'std': float(np.std(e_xy)), 'min': float(np.min(e_xy)), 'max': float(np.max(e_xy))} if e_xy else None,
            'landing_time': {'mean': float(np.mean(lt)), 'std': float(np.std(lt)), 'min': float(np.min(lt)), 'max': float(np.max(lt))} if lt else None,
            'waiting_time': {'mean': float(np.mean(wt)), 'std': float(np.std(wt)), 'min': float(np.min(wt)), 'max': float(np.max(wt))} if wt else None,
        }

    data = {
        'total_experiments': len(results),
        'timestamp': datetime.now().isoformat(),
        'experiments': experiments,
        'statistics': statistics,
    }
    path = os.path.join(base_dir, 'results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Saved %s", path)


def main():
    parser = argparse.ArgumentParser(description='重复 x 次降落实验，统计降落误差，仅保存 JSON')
    parser.add_argument('--num', type=int, default=10, help='重复次数')
    parser.add_argument('--output-dir', type=str, default=None, help='结果目录，默认带时间戳')
    parser.add_argument('--wait-min', type=float, default=2.0, help='触发前等待时间下限 (s)')
    parser.add_argument('--wait-max', type=float, default=10.0, help='触发前等待时间上限 (s)')
    parser.add_argument('--no-rosbag', action='store_true', help='不录 rosbag')
    args = parser.parse_args()


    args.num = 10
    args.wait_min = 2.0
    args.wait_max = 10.0
    args.no_rosbag = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_landing_repeat', anonymous=True)

    base_dir = args.output_dir
    if not base_dir:
        base_dir = os.path.join(WORKSPACE, f'experiment_landing_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(base_dir, exist_ok=True)
    rospy.loginfo("Output directory: %s", base_dir)

    trigger_pub = rospy.Publisher('/mode_manager', PoseStamped, queue_size=10)
    pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=10)
    uav_listener = UAVPositionListener()
    target_listener = TargetPositionListener()
    time.sleep(2)

    results = []
    for i in range(1, args.num + 1):
        if shutdown_flag.is_set():
            rospy.logwarn("Shutdown flag set. Stopping.")
            break
        try:
            r = run_single_experiment(
                i, args.num,
                uav_listener, target_listener,
                trigger_pub, pos_cmd_pub,
                base_dir,
                wait_min=args.wait_min,
                wait_max=args.wait_max,
                record_rosbag=not args.no_rosbag,
            )
            if r is None:
                break
            results.append(r)
            if i % 10 == 0:
                save_results(results, base_dir)
        except KeyboardInterrupt:
            rospy.logwarn("KeyboardInterrupt. Saving and exiting.")
            break
        except Exception as e:
            rospy.logerr("Experiment %d failed: %s", i, e)
            results.append({
                'experiment_num': i,
                'waiting_time': None,
                'landing_time': None,
                'success': False,
                'error_x': None,
                'error_y': None,
                'error_xy': None,
                'uav_pos': None,
                'target_pos': None,
                'stopped': True,
                'exp_dir': os.path.join(base_dir, f"experiment_{i:03d}"),
            })

    cleanup_processes()
    if results:
        save_results(results, base_dir)
    rospy.loginfo("Done. Total runs: %d. Results in %s", len(results), base_dir)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
