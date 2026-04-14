#!/usr/bin/env python3
"""
实验：dog_pos 的 yaw 漂移速率与 offset 漂移速率（同一参数同步增加）与定位误差的关系
流程：
0. 需先启动 roscore
1. 清理之前相关参数
2. 设置参数：漂移速率从 drift_min 到 drift_max 均匀取 drift_num 个点（该参数同时作为 yaw 漂移速率 rad/s 与 offset 漂移速率 m/s 使用，同步增加）
3. 启动 roslaunch planning dog_pos_processor.launch 和 python3 fake_target.py
4. 记录 dog_pos_processed 与 ground_truth_traj 的误差（ground_truth 收到时计算，dog_pos_processed 插值），持续 duration 秒
5. 关闭两个进程，保存该次平均误差；回到步骤 1 直到跑完 drift_num 次
6. 将漂移速率序列与平均误差保存为 JSON，并绘制曲线图（x=漂移速率，y=平均误差）
支持 Ctrl+C 安全退出并保存已得结果。
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

import rospy
from nav_msgs.msg import Odometry

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
FAKE_TARGET_SCRIPT = os.path.join(WORKSPACE, 'fake_target.py')
SETUP_BASH = os.path.join(WORKSPACE, 'devel', 'setup.bash')

active_processes = {'launch': None, 'fake_target': None}
active_processes_lock = threading.Lock()
shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    rospy.logwarn("Received interrupt (Ctrl+C). Shutting down safely...")
    shutdown_flag.set()
    cleanup_processes()


def cleanup_processes():
    with active_processes_lock:
        for name in ['fake_target', 'launch']:
            proc = active_processes.get(name)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception as e:
                    rospy.logwarn("Error stopping %s: %s", name, str(e))
                active_processes[name] = None
    rospy.loginfo("Processes cleaned up.")


def clear_target_circle_drift_params():
    """清空 target_circle_publisher 下漂移相关参数"""
    ns = '/target_circle_publisher'
    params = [
        'dog_pos_drift_offset_x', 'dog_pos_drift_offset_y', 'dog_pos_drift_offset_z',
        'dog_pos_drift_rate_x', 'dog_pos_drift_rate_y', 'dog_pos_drift_rate_z',
        'dog_pos_drift_yaw_initial', 'dog_pos_drift_yaw_rate',
    ]
    for p in params:
        try:
            full = f'{ns}/{p}'
            if rospy.has_param(full):
                rospy.delete_param(full)
        except Exception:
            pass


class ErrorRecorder:
    """订阅 ground_truth_traj 和 dog_pos_processed，在每次收到 ground_truth 时用插值计算误差并记录"""

    def __init__(self):
        self.dog_pos_buffer = deque(maxlen=5000)
        self.errors = []
        self.ready = False  # 是否已经 pos/yaw ready
        self.lock = threading.Lock()
        self._gt_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self._gt_cb, queue_size=50)
        self._dog_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self._dog_cb, queue_size=50)

    def _dog_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        with self.lock:
            # dog_pos_processor.cpp 中约定：
            # orientation.w = precise_pos_offset_ready_ ? 1.0 : 0.0
            # orientation.x = precise_yaw_offset_ready_ ? 1.0 : 0.0
            pos_ready = msg.pose.pose.orientation.w >= 0.5
            yaw_ready = msg.pose.pose.orientation.x >= 0.5
            ready_now = pos_ready and yaw_ready
            if ready_now and not self.ready:
                # 第一次 ready，清空历史缓存，从稳定后开始记录
                self.errors = []
                self.dog_pos_buffer.clear()
            self.ready = ready_now
            self.dog_pos_buffer.append((t, x, y, z))

    def _interpolate(self, t):
        with self.lock:
            buf = list(self.dog_pos_buffer)
        if not buf:
            return None
        buf.sort(key=lambda e: e[0])
        ts = np.array([e[0] for e in buf])
        if t <= ts[0]:
            return (buf[0][1], buf[0][2], buf[0][3])
        if t >= ts[-1]:
            return (buf[-1][1], buf[-1][2], buf[-1][3])
        i = np.searchsorted(ts, t)
        t0, x0, y0, z0 = buf[i - 1][0], buf[i - 1][1], buf[i - 1][2], buf[i - 1][3]
        t1, x1, y1, z1 = buf[i][0], buf[i][1], buf[i][2], buf[i][3]
        if t1 == t0:
            return (x0, y0, z0)
        alpha = (t - t0) / (t1 - t0)
        return (
            x0 + alpha * (x1 - x0),
            y0 + alpha * (y1 - y0),
            z0 + alpha * (z1 - z0),
        )

    def _gt_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y
        z_gt = msg.pose.pose.position.z
        with self.lock:
            if not self.ready:
                return
        pos = self._interpolate(t)
        if pos is None:
            return
        err = math.sqrt((x_gt - pos[0]) ** 2 + (y_gt - pos[1]) ** 2 + (z_gt - pos[2]) ** 2)
        with self.lock:
            self.errors.append(err)

    def get_mean_error_and_reset(self):
        with self.lock:
            errs = list(self.errors)
            self.errors = []
            self.dog_pos_buffer.clear()
            self.ready = False
        return float(np.mean(errs)) if errs else float('nan')

    def reset(self):
        with self.lock:
            self.errors = []
            self.dog_pos_buffer.clear()
            self.ready = False


def run_single_trial(drift_rate, duration_sec, recorder, workspace):
    """单次实验：设置漂移参数（yaw 与 offset 同步为该速率）-> 启动 launch 与 fake_target -> 记录 duration_sec -> 关闭 -> 返回平均误差"""
    if shutdown_flag.is_set():
        return None

    clear_target_circle_drift_params()
    time.sleep(0.2)
    ns = '/target_circle_publisher'
    # 同一参数同步增加：yaw 漂移速率 (rad/s) 与 offset 漂移速率 (m/s) 均设为 drift_rate
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', float(drift_rate))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_x', float(drift_rate))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_y', float(drift_rate))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    recorder.reset()

    # 先启动 fake_target，等 1s 后再启动 dog_pos_processor
    env = os.environ.copy()
    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=3)
            except Exception:
                pass
            active_processes['fake_target'] = None
        active_processes['fake_target'] = subprocess.Popen(
            [sys.executable, FAKE_TARGET_SCRIPT],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    time.sleep(1.0)  # 运行 fake_target 后等 1s 再运行 dog_pos_processor
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    cmd_launch = f'source "{SETUP_BASH}" && roslaunch planning dog_pos_processor.launch'
    with active_processes_lock:
        if active_processes['launch'] is not None:
            try:
                active_processes['launch'].terminate()
                active_processes['launch'].wait(timeout=3)
            except Exception:
                pass
            active_processes['launch'] = None
        active_processes['launch'] = subprocess.Popen(
            ['bash', '-c', cmd_launch],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    time.sleep(2.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 硬延迟 2s 后再开始统计
    time.sleep(2.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    recorder.reset()
    elapsed = 0.0
    step = 0.1
    while elapsed < duration_sec and not shutdown_flag.is_set():
        time.sleep(step)
        elapsed += step
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    mean_err = recorder.get_mean_error_and_reset()
    cleanup_processes()
    time.sleep(1.0)
    return mean_err


def main():
    parser = argparse.ArgumentParser(description='dog_pos yaw/offset 漂移速率（同步）与定位误差关系')
    parser.add_argument('--drift-min', type=float, default=0.0, help='漂移速率下限（yaw: rad/s, offset: m/s 同值）')
    parser.add_argument('--drift-max', type=float, default=0.05, help='漂移速率上限')
    parser.add_argument('--drift-num', type=int, default=11, help='漂移速率采样点数')
    parser.add_argument('--duration', type=float, default=10.0, help='每次实验记录时长 (s)')
    parser.add_argument('--output-dir', type=str, default=None, help='结果输出目录')
    args = parser.parse_args()

    args.drift_min = 0.0
    args.drift_max = 0.1
    args.drift_num = 10
    args.duration = 10.0

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_dog_pos_drift_error', anonymous=True)

    workspace = WORKSPACE
    if not os.path.isfile(SETUP_BASH):
        rospy.logerr("Not found: %s. Run from catkin workspace.", SETUP_BASH)
        sys.exit(1)
    if not os.path.isfile(FAKE_TARGET_SCRIPT):
        rospy.logerr("Not found: %s", FAKE_TARGET_SCRIPT)
        sys.exit(1)

    drift_vals = np.linspace(args.drift_min, args.drift_max, args.drift_num).tolist()
    if args.drift_num == 1:
        drift_vals = [args.drift_min]

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(workspace, f'experiment_dog_pos_drift_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    rospy.loginfo("Output directory: %s", out_dir)

    recorder = ErrorRecorder()
    time.sleep(1.0)

    errors = []
    for i, drift in enumerate(drift_vals):
        if shutdown_flag.is_set():
            break
        rospy.loginfo("Trial %d/%d: drift_rate=%.4f (yaw rad/s & offset m/s)", i + 1, len(drift_vals), drift)
        mean_err = run_single_trial(drift, args.duration, recorder, workspace)
        if mean_err is None:
            errors.append(float('nan'))
        else:
            errors.append(mean_err)
            rospy.loginfo("  mean error = %.4f m", mean_err)

    while len(errors) < len(drift_vals):
        errors.append(float('nan'))

    def to_serializable(obj):
        if isinstance(obj, (list, np.ndarray)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if np.isfinite(v) else None
        return obj

    data = {
        'drift_rate': to_serializable(drift_vals),
        'mean_error_m': to_serializable(errors),
        'duration_sec': args.duration,
        'note': 'drift_rate: same value for dog_pos_drift_yaw_rate (rad/s) and dog_pos_drift_rate_xy (m/s)',
        'timestamp': datetime.now().isoformat(),
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Saved %s", json_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        drift_arr = np.array(drift_vals)
        err_arr = np.array(errors)
        valid = np.isfinite(err_arr)
        if np.any(valid):
            fig, ax = plt.subplots()
            ax.plot(drift_arr, err_arr, 'o-', markersize=6)
            ax.set_xlabel('Drift rate (yaw: rad/s, offset: m/s, same value)')
            ax.set_ylabel('Mean positioning error (m)')
            ax.set_title('Dog pos drift rate vs positioning error')
            ax.grid(True, alpha=0.3)
            curve_path = os.path.join(out_dir, 'curve.png')
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            rospy.loginfo("Saved curve %s", curve_path)
    except Exception as e:
        rospy.logwarn("Could not save curve: %s", str(e))

    rospy.loginfo("Experiment finished. Results in %s", out_dir)


if __name__ == '__main__':
    main()
