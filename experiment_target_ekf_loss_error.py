#!/usr/bin/env python3
"""
实验：target_ekf_odom 丢失后的平均定位误差与 dog_pos 漂移速率的关系（曲线：drift vs mean error after loss）
流程：
0. 需先启动 roscore
1. 清理漂移及 target 丢失相关参数
2. 漂移类型 drift_type：yaw / pos / both；漂移速率从 drift_min 到 drift_max 取 drift_num 个点
3. 每次 trial：设置漂移（不预先设 stop 参数），启动 fake_target + dog_pos_processor
4. 硬延迟 2s 后等待 dog_pos_processed 收敛（pos/yaw ready）
5. 收敛后再设置 target_ekf_odom_stop_after_sec=loss_after_sec；fake_target 从「首次读到该参数」起计时，N 秒后停止发布
6. 再等 loss_after_sec 秒后开始记录 duration 秒内误差，取平均，得到「丢失后的平均误差」
7. 保存 (drift_vals, mean_errors) 到 JSON，并绘制曲线图
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


def clear_freq_and_drift_params():
    """清空 target_circle_publisher 下频率、漂移及 target 丢失相关参数"""
    ns = '/target_circle_publisher'
    params = [
        'target_ekf_odom_hz', 'dog_pos_hz', 'target_ekf_odom_noise_std', 'dog_pos_noise_std',
        'target_ekf_odom_stop_after_sec',
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
    """订阅 ground_truth_traj 和 dog_pos_processed，在每次收到 ground_truth 时用插值计算误差并记录；可等收敛(ready)后再记"""

    def __init__(self, use_ready=True):
        self.dog_pos_buffer = deque(maxlen=5000)
        self.errors = []
        self.use_ready = use_ready
        self.ready = False
        self.lock = threading.Lock()
        self._gt_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self._gt_cb, queue_size=50)
        self._dog_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self._dog_cb, queue_size=50)

    def is_ready(self):
        with self.lock:
            return self.ready

    def _dog_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        with self.lock:
            if self.use_ready:
                pos_ready = msg.pose.pose.orientation.w >= 0.5
                yaw_ready = msg.pose.pose.orientation.x >= 0.5
                ready_now = pos_ready and yaw_ready
                if ready_now and not self.ready:
                    self.errors = []
                    self.dog_pos_buffer.clear()
                self.ready = ready_now
            else:
                self.ready = True
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
        with self.lock:
            if self.use_ready and not self.ready:
                return
        t = msg.header.stamp.to_sec()
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y
        z_gt = msg.pose.pose.position.z
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
            self.ready = False if self.use_ready else True
        return float(np.mean(errs)) if errs else float('nan')

    def reset(self):
        with self.lock:
            self.errors = []
            self.dog_pos_buffer.clear()
            self.ready = False if self.use_ready else True


def run_single_trial(drift_rate, duration_sec, recorder, workspace, loss_after_sec,
                    drift_type='both', yaw_drift_rate=None):
    """
    单次实验：设置漂移 -> 启动 fake_target + dog_pos_processor -> 硬延迟 2s -> 等收敛
    -> 收敛后设置 target_ekf_odom_stop_after_sec=loss_after_sec（从此时起再过 loss_after_sec 秒停发）
    -> 等 loss_after_sec 秒后开始统计「丢失后的平均误差」duration_sec 秒。
    """
    if shutdown_flag.is_set():
        return None

    clear_freq_and_drift_params()
    time.sleep(0.2)
    ns = '/target_circle_publisher'

    rospy.set_param(f'{ns}/target_ekf_odom_hz', 15.0)
    rospy.set_param(f'{ns}/dog_pos_hz', 50.0)
    # target_ekf_odom_stop_after_sec 在收敛后再设置，fake_target 从「首次读到该参数」起计时
    yaw_val = float(yaw_drift_rate if (drift_type == 'both' and yaw_drift_rate is not None) else drift_rate)
    if drift_type in ('both', 'yaw'):
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', yaw_val)
    else:
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', 0.0)
    if drift_type in ('both', 'pos'):
        rospy.set_param(f'{ns}/dog_pos_drift_rate_x', float(drift_rate))
        rospy.set_param(f'{ns}/dog_pos_drift_rate_y', float(drift_rate))
    else:
        rospy.set_param(f'{ns}/dog_pos_drift_rate_x', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    recorder.reset()

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
    time.sleep(1.0)
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

    # 等待 dog_pos_processed 收敛（pos/yaw ready）
    rospy.loginfo("Waiting for dog_pos_processed pos/yaw ready...")
    while not shutdown_flag.is_set():
        if recorder.is_ready():
            rospy.loginfo("Ready. Setting target_ekf_odom_stop_after_sec=%.1f (stop in %.1f s from now).", loss_after_sec, loss_after_sec)
            rospy.set_param(f'{ns}/target_ekf_odom_stop_after_sec', float(loss_after_sec))
            break
        time.sleep(0.05)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 等 loss_after_sec 秒后 fake_target 实际停发 target_ekf_odom
    time.sleep(loss_after_sec)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    recorder.reset()  # 只统计丢失之后的误差
    rospy.loginfo("Target loss reached. Recording errors for %.1f s.", duration_sec)

    start_time = time.time()
    elapsed = 0.0
    step = 0.1
    while elapsed < duration_sec and not shutdown_flag.is_set():
        time.sleep(step)
        elapsed = time.time() - start_time
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    mean_err = recorder.get_mean_error_and_reset()
    cleanup_processes()
    time.sleep(1.0)
    return mean_err


def main():
    parser = argparse.ArgumentParser(description='target 丢失后的平均定位误差与漂移速率关系（曲线）')
    parser.add_argument('--loss-after', type=float, default=60.0, help='收敛后再过多少秒停止发布 target_ekf_odom（丢失时刻）')
    parser.add_argument('--duration', type=float, default=10.0, help='丢失后记录误差的时长 (s)')
    parser.add_argument('--drift-min', type=float, default=0.0, help='漂移速率下限（rad/s 或 m/s，依 drift_type）')
    parser.add_argument('--drift-max', type=float, default=0.1, help='漂移速率上限')
    parser.add_argument('--drift-num', type=int, default=6, help='漂移采样点数')
    parser.add_argument(
        '--drift-type',
        type=str,
        default='both',
        choices=['both', 'yaw', 'pos'],
        help='漂移类型：both=同时 yaw_rate 和 pos_rate(x/y)，yaw=仅 yaw_rate，pos=仅 pos_rate',
    )
    parser.add_argument('--yaw-drift-min', type=float, default=None,
                        help='drift_type=both 时 yaw 漂移下限 (rad/s)，未设则用 drift-min')
    parser.add_argument('--yaw-drift-max', type=float, default=None,
                        help='drift_type=both 时 yaw 漂移上限 (rad/s)，未设则用 drift-max')
    parser.add_argument('--output-dir', type=str, default=None, help='结果输出目录')
    args = parser.parse_args()

    args.drift_type = 'both'
    args.yaw_drift_min = 0.0
    args.yaw_drift_max = 0.01


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_target_ekf_loss_error', anonymous=True)

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

    if args.drift_type == 'both':
        yaw_drift_min = args.yaw_drift_min if args.yaw_drift_min is not None else args.drift_min
        yaw_drift_max = args.yaw_drift_max if args.yaw_drift_max is not None else args.drift_max
    else:
        yaw_drift_min = yaw_drift_max = None

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(workspace, f'experiment_drift_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    rospy.loginfo("Output directory: %s", out_dir)
    rospy.loginfo("loss_after=%.1fs, duration=%.1fs, drift_type=%s, drift_vals=%s",
                  args.loss_after, args.duration, args.drift_type, drift_vals)

    recorder = ErrorRecorder()
    time.sleep(1.0)

    mean_errors = []
    for j, drift in enumerate(drift_vals):
        if shutdown_flag.is_set():
            break
        if args.drift_type == 'both' and args.drift_num > 1:
            t = (drift - args.drift_min) / (args.drift_max - args.drift_min)
            yaw_drift = yaw_drift_min + t * (yaw_drift_max - yaw_drift_min)
        elif args.drift_type == 'both':
            yaw_drift = yaw_drift_min
        else:
            yaw_drift = None
        rospy.loginfo("Trial %d/%d: drift=%.4f, type=%s%s",
                      j + 1, len(drift_vals), drift, args.drift_type,
                      f", yaw_drift=%.4f" % yaw_drift if yaw_drift is not None else "")
        mean_err = run_single_trial(
            drift,
            args.duration,
            recorder,
            workspace,
            args.loss_after,
            drift_type=args.drift_type,
            yaw_drift_rate=yaw_drift,
        )
        if mean_err is None:
            mean_errors.append(float('nan'))
        else:
            mean_errors.append(mean_err)
            rospy.loginfo("  mean error = %.4f m", mean_err)

    def to_serializable(obj):
        if isinstance(obj, (list, np.ndarray)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    data = {
        'loss_after_sec': args.loss_after,
        'duration_sec': args.duration,
        'drift_type': args.drift_type,
        'drift_rate': to_serializable(drift_vals),
        'mean_error_m': to_serializable(mean_errors),
        'note': 'mean_error_m[i] = mean positioning error (m) AFTER target_ekf_odom loss, at drift_rate[i].',
        'timestamp': datetime.now().isoformat(),
    }
    if args.drift_type == 'both':
        data['yaw_drift_min'] = yaw_drift_min
        data['yaw_drift_max'] = yaw_drift_max
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Saved %s", json_path)

    # 曲线：横轴 drift，纵轴 mean error
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d_arr = np.array(drift_vals)
        e_arr = np.array(mean_errors)
        valid = np.isfinite(e_arr)
        if np.any(valid):
            fig, ax = plt.subplots()
            ax.plot(d_arr, e_arr, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Drift rate (rad/s or m/s)', fontweight='bold')
            ax.set_ylabel('Mean positioning error (m)', fontweight='bold')
            ax.set_title('Mean error after loss vs drift (loss_after=%.1fs, duration %.1fs)' % (args.loss_after, args.duration))
            ax.grid(True, alpha=0.3)
            curve_path = os.path.join(out_dir, 'curve.png')
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            rospy.loginfo("Saved curve %s", curve_path)
    except Exception as ex:
        rospy.logwarn("Could not save curve: %s", str(ex))

    rospy.loginfo("Experiment finished. Results in %s", out_dir)


if __name__ == '__main__':
    main()
