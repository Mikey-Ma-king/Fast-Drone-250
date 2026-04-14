#!/usr/bin/env python3
"""
实验：target_ekf_odom 频率与 dog_pos 初始漂移对定位误差的影响（热力图）
流程：
0. 需先启动 roscore
1. 清理之前相关参数
2. 设置参数：频率 a~b 取 x 个点，初始漂移 c~d 取 y 个点
   - target_ekf_odom_hz 扫频
   - dog_pos_drift_yaw_initial (rad)
   - dog_pos_drift_offset_x/y (m)
   可选三种模式：
     * both：yaw_initial 和 offset_x/y 同时 initial drift
     * yaw：仅 yaw_initial 有 initial drift，offset_x/y = 0
     * pos：仅 offset_x/y 有 initial drift，yaw_initial = 0
   所有“漂移速率”参数设为 0
3. 启动 roslaunch planning dog_pos_processor.launch 和 python3 fake_target.py（均先 source devel/setup.bash）
4. 硬延迟 2s 后记录 dog_pos_processed 与 ground_truth_traj 的误差，持续 duration 秒
5. 关闭两个进程，保存该次平均误差；回到步骤 1 直到跑完 x*y 次
6. 将平均误差矩阵及频率、初始漂移设置保存为 JSON，并绘制热力图（左 frequency，下 initial_drift，右 error，标签加粗）
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
    """清空 target_circle_publisher 下频率与漂移相关参数"""
    ns = '/target_circle_publisher'
    params = [
        'target_ekf_odom_hz', 'dog_pos_hz', 'target_ekf_odom_noise_std', 'dog_pos_noise_std',
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

    def __init__(self, use_ready=True):
        self.dog_pos_buffer = deque(maxlen=5000)
        self.errors = []
        self.use_ready = use_ready  # 是否根据 pos/yaw ready 决定何时开始记录
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
            if self.use_ready:
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
            else:
                # 不使用 ready 机制时，始终视为 ready
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
        t = msg.header.stamp.to_sec()
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y
        z_gt = msg.pose.pose.position.z
        with self.lock:
            if self.use_ready and not self.ready:
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
            self.ready = False if self.use_ready else True
        return float(np.mean(errs)) if errs else float('nan')

    def reset(self):
        with self.lock:
            self.errors = []
            self.dog_pos_buffer.clear()
            self.ready = False if self.use_ready else True

    def is_ready(self):
        with self.lock:
            return self.ready


def run_single_trial(freq_hz, initial_drift, duration_sec, recorder, workspace, drift_type='both', wait_pos_ready=False):
    """单次实验：设置频率与初始漂移参数（可选仅 yaw / 仅 pos / 同步）-> 启动 launch 与 fake_target
    -> （可选）等待 dog_pos_processed pos/yaw ready 收敛后再开始 duration 计时 -> 记录 duration_sec -> 关闭 -> 返回平均误差"""
    if shutdown_flag.is_set():
        return None

    clear_freq_and_drift_params()
    time.sleep(0.2)
    ns = '/target_circle_publisher'

    # wait_pos_ready 时：在 ready 之前 freq 用默认 15、drift 保持为 0，等 ready 之后再下发给定参数
    if wait_pos_ready:
        rospy.set_param(f'{ns}/target_ekf_odom_hz', 15.0)
        rospy.set_param(f'{ns}/dog_pos_hz', 50.0)
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_x', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_y', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    else:
        rospy.set_param(f'{ns}/target_ekf_odom_hz', float(freq_hz))
        rospy.set_param(f'{ns}/dog_pos_hz', 50.0)
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_x', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_y', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
        if drift_type in ('both', 'yaw'):
            rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', float(initial_drift))
        else:
            rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
        if drift_type in ('both', 'pos'):
            rospy.set_param(f'{ns}/dog_pos_drift_offset_x', float(initial_drift))
            rospy.set_param(f'{ns}/dog_pos_drift_offset_y', float(initial_drift))
        else:
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

    # 硬延迟 2s 后再开始统计 / 等待 pos ready
    time.sleep(2.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    recorder.reset()

    # 可选：等待 dog_pos_processed 的 pos/yaw ready，ready 之后再下发 freq 和 drift 参数
    if wait_pos_ready:
        rospy.loginfo("Waiting for dog_pos_processed pos/yaw ready before starting timer...")
        while not shutdown_flag.is_set():
            if recorder.is_ready():
                rospy.loginfo("dog_pos_processed ready. Setting freq and drift params, then start timing.")
                break
            time.sleep(0.05)
        if shutdown_flag.is_set():
            cleanup_processes()
            return None
        # ready 之后再给 freq 和 drift 的 parameter 指令
        rospy.set_param(f'{ns}/target_ekf_odom_hz', float(freq_hz))
        rospy.set_param(f'{ns}/dog_pos_hz', 50.0)
        rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_x', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_y', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
        if drift_type in ('both', 'yaw'):
            rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', float(initial_drift))
        else:
            rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
        if drift_type in ('both', 'pos'):
            rospy.set_param(f'{ns}/dog_pos_drift_offset_x', float(initial_drift))
            rospy.set_param(f'{ns}/dog_pos_drift_offset_y', float(initial_drift))
        else:
            rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
            rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
        rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
        time.sleep(0.2)  # 给参数生效一点时间

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
    parser = argparse.ArgumentParser(description='Target EKF 频率与 dog_pos 初始漂移对定位误差影响（热力图）')
    parser.add_argument('--freq-min', type=float, default=5.0, help='target_ekf_odom 频率下限 (Hz)')
    parser.add_argument('--freq-max', type=float, default=30.0, help='target_ekf_odom 频率上限 (Hz)')
    parser.add_argument('--freq-num', type=int, default=10, help='频率采样点数')
    parser.add_argument('--drift-min', type=float, default=0.0, help='dog_pos 初始漂移下限（yaw_initial: rad & offset_x/y: m 同值）')
    parser.add_argument('--drift-max', type=float, default=3.0, help='初始漂移上限')
    parser.add_argument('--drift-num', type=int, default=10, help='初始漂移采样点数')
    parser.add_argument(
        '--drift-type',
        type=str,
        default='pos',
        choices=['both', 'yaw', 'pos'],
        help='初始漂移类型：both=同时 yaw_initial 和 offset_x/y，yaw=仅 yaw_initial，pos=仅 offset_x/y',
    )
    parser.add_argument(
        '--wait-pos-ready',
        action='store_true',
        help='是否在计时前等待 dog_pos_processed 的 pos/yaw ready（等收敛后再开始 duration 计时）',
    )
    parser.add_argument('--duration', type=float, default=10.0, help='每次实验记录时长 (s)')
    parser.add_argument('--output-dir', type=str, default=None, help='结果输出目录')
    args = parser.parse_args()

    args.wait_pos_ready = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_target_ekf_freq_initial_drift', anonymous=True)

    workspace = WORKSPACE
    if not os.path.isfile(SETUP_BASH):
        rospy.logerr("Not found: %s. Run from catkin workspace.", SETUP_BASH)
        sys.exit(1)
    if not os.path.isfile(FAKE_TARGET_SCRIPT):
        rospy.logerr("Not found: %s", FAKE_TARGET_SCRIPT)
        sys.exit(1)

    freq_vals = np.linspace(args.freq_min, args.freq_max, args.freq_num).tolist()
    drift_vals = np.linspace(args.drift_min, args.drift_max, args.drift_num).tolist()
    if args.freq_num == 1:
        freq_vals = [args.freq_min]
    if args.drift_num == 1:
        drift_vals = [args.drift_min]

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(workspace, f'experiment_freq_initial_drift_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    rospy.loginfo("Output directory: %s", out_dir)

    recorder = ErrorRecorder(use_ready=args.wait_pos_ready)
    time.sleep(1.0)

    # error_matrix[i][j] = 频率 freq_vals[i], 初始漂移 drift_vals[j] 时的平均误差（行=freq，列=initial_drift）
    error_matrix = []
    total = len(freq_vals) * len(drift_vals)
    idx = 0

    for fi, freq in enumerate(freq_vals):
        row = []
        for dj, drift in enumerate(drift_vals):
            idx += 1
            if shutdown_flag.is_set():
                break
            rospy.loginfo(
                "Trial %d/%d: freq=%.2f Hz, initial_drift=%.4f, type=%s, wait_pos_ready=%s",
                idx, total, freq, drift, args.drift_type, args.wait_pos_ready,
            )
            mean_err = run_single_trial(
                freq,
                drift,
                args.duration,
                recorder,
                workspace,
                args.drift_type,
                args.wait_pos_ready,
            )
            if mean_err is None:
                row.append(float('nan'))
            else:
                row.append(mean_err)
                rospy.loginfo("  mean error = %.4f m", mean_err)
        if shutdown_flag.is_set():
            break
        error_matrix.append(row)

    while len(error_matrix) < len(freq_vals):
        error_matrix.append([float('nan')] * len(drift_vals))
    for row in error_matrix:
        while len(row) < len(drift_vals):
            row.append(float('nan'))

    def to_serializable(obj):
        if isinstance(obj, (list, np.ndarray)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if np.isfinite(v) else None
        return obj

    data = {
        'freq_Hz': to_serializable(freq_vals),
        'initial_drift': to_serializable(drift_vals),
        'duration_sec': args.duration,
        'error_matrix': to_serializable(error_matrix),
        'note': (
            'error_matrix[i][j] = mean error at freq_vals[i], initial_drift_vals[j]; '
            f'initial_drift_type={args.drift_type}; yaw_initial 和/或 offset_x/y 按类型设置，所有 drift_rate = 0.'
        ),
        'timestamp': datetime.now().isoformat(),
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Saved %s", json_path)

    # 热力图：横轴 initial_drift，纵轴 frequency，颜色 error，带单位，双线性插值平滑过渡
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        M = np.array(error_matrix)
        if M.size > 0 and np.any(np.isfinite(M)):
            fig, ax = plt.subplots()
            im = ax.imshow(
                M,
                extent=[drift_vals[0], drift_vals[-1], freq_vals[0], freq_vals[-1]],
                aspect='auto',
                cmap='viridis',
                origin='lower',
                interpolation='bilinear',
            )
            ax.set_xlabel('initial drift (rad or m, see drift_type)', fontweight='bold')
            ax.set_ylabel('frequency (Hz)', fontweight='bold')
            ax.set_title('Mean positioning error (m) vs target_ekf_odom freq & dog_pos initial drift')
            cbar = plt.colorbar(im, ax=ax, label='error (m)')
            cbar.set_label('error (m)', fontweight='bold')
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
            cbar.ax.yaxis.get_label().set_fontweight('bold')
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_fontweight('bold')
            heatmap_path = os.path.join(out_dir, 'heatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            rospy.loginfo("Saved heatmap %s", heatmap_path)
    except Exception as e:
        rospy.logwarn("Could not save heatmap: %s", str(e))

    rospy.loginfo("Experiment finished. Results in %s", out_dir)


if __name__ == '__main__':
    main()

