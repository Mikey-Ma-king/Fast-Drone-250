#!/usr/bin/env python3
"""
实验：target_ekf_odom 频率与噪声对定位误差的影响
流程：
0. 需先启动 roscore
1. 清理之前相关参数
2. 设置参数（频率 a~b 取 x 个点，噪声 c~d 取 y 个点）
3. 启动 roslaunch planning dog_pos_processor.launch 和 python3 fake_target.py（均先 source devel/setup.bash）
4. 记录 dog_pos_processed 与 ground_truth_traj 的误差（每次收到 ground_truth 时计算，dog_pos_processed 插值），持续 duration 秒
5. 关闭两个进程，保存该次平均误差；回到步骤 1 直到跑完 x*y 次
6. 将平均误差矩阵及频率、噪声设置保存为 JSON，并绘制热力图
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

# ROS
import rospy
from nav_msgs.msg import Odometry

# 工作空间路径（脚本所在目录的上一级为 Fast-Drone-250）
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
FAKE_TARGET_SCRIPT = os.path.join(WORKSPACE, 'fake_target.py')
SETUP_BASH = os.path.join(WORKSPACE, 'devel', 'setup.bash')

# 子进程
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


def clear_target_circle_params():
    """清空 target_circle_publisher 相关参数（便于下次覆盖）"""
    ns = '/target_circle_publisher'
    params = ['target_ekf_odom_hz', 'dog_pos_hz', 'target_ekf_odom_noise_std', 'dog_pos_noise_std']
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
        self.dog_pos_buffer = deque(maxlen=5000)  # (stamp_sec, x, y, z)
        self.errors = []
        self.lock = threading.Lock()
        self._gt_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self._gt_cb, queue_size=50)
        self._dog_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self._dog_cb, queue_size=50)

    def _dog_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        with self.lock:
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
        return float(np.mean(errs)) if errs else float('nan')

    def reset(self):
        with self.lock:
            self.errors = []
            self.dog_pos_buffer.clear()


def run_single_trial(freq_hz, noise_std, duration_sec, recorder, workspace):
    """单次实验：设置参数 -> 启动 launch 与 fake_target -> 记录 duration_sec -> 关闭 -> 返回平均误差"""
    if shutdown_flag.is_set():
        return None

    # 1. 清理并设置参数（在启动 fake_target 前设置，节点会读取）
    clear_target_circle_params()
    time.sleep(0.2)
    rospy.set_param('/target_circle_publisher/target_ekf_odom_hz', float(freq_hz))
    rospy.set_param('/target_circle_publisher/target_ekf_odom_noise_std', float(noise_std))
    # 可选：固定 dog_pos 频率
    rospy.set_param('/target_circle_publisher/dog_pos_hz', 50.0)
    recorder.reset()

    # 2. 先启动 fake_target，等 1s 后再启动 dog_pos_processor
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

    # 3. 启动 dog_pos_processor (roslaunch)
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

    # 4. 硬延迟 2s 后再开始统计
    time.sleep(2.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    recorder.reset()
    # 5. 记录 duration_sec
    elapsed = 0.0
    step = 0.1
    while elapsed < duration_sec and not shutdown_flag.is_set():
        time.sleep(step)
        elapsed += step
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    mean_err = recorder.get_mean_error_and_reset()

    # 6. 关掉两个进程
    cleanup_processes()
    time.sleep(1.0)

    return mean_err


def main():
    parser = argparse.ArgumentParser(description='Target EKF 频率与噪声对定位误差影响的网格实验')
    parser.add_argument('--freq-min', type=float, default=5.0, help='target_ekf_odom 频率下限 (Hz)')
    parser.add_argument('--freq-max', type=float, default=30.0, help='target_ekf_odom 频率上限 (Hz)')
    parser.add_argument('--freq-num', type=int, default=6, help='频率采样点数')
    parser.add_argument('--noise-min', type=float, default=0.0, help='噪声标准差下限 (m)')
    parser.add_argument('--noise-max', type=float, default=0.2, help='噪声标准差上限 (m)')
    parser.add_argument('--noise-num', type=int, default=5, help='噪声采样点数')
    parser.add_argument('--duration', type=float, default=10.0, help='每次实验记录时长 (s)')
    parser.add_argument('--output-dir', type=str, default=None, help='结果输出目录，默认 workspace 下带时间戳目录')
    args = parser.parse_args()


    args.freq_min = 5.0
    args.freq_max = 25.0
    args.freq_num = 10
    args.noise_min = 0.0
    args.noise_max = 0.25
    args.noise_num = 10
    args.duration = 10.0


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_target_ekf_freq_noise', anonymous=True)

    workspace = WORKSPACE
    if not os.path.isfile(SETUP_BASH):
        rospy.logerr("Not found: %s. Run from catkin workspace.", SETUP_BASH)
        sys.exit(1)
    if not os.path.isfile(FAKE_TARGET_SCRIPT):
        rospy.logerr("Not found: %s", FAKE_TARGET_SCRIPT)
        sys.exit(1)

    freq_vals = np.linspace(args.freq_min, args.freq_max, args.freq_num).tolist()
    noise_vals = np.linspace(args.noise_min, args.noise_max, args.noise_num).tolist()
    if args.noise_num == 1:
        noise_vals = [args.noise_min]
    if args.freq_num == 1:
        freq_vals = [args.freq_min]

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(workspace, f'experiment_target_ekf_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    rospy.loginfo("Output directory: %s", out_dir)

    recorder = ErrorRecorder()
    time.sleep(1.0)

    # 误差矩阵: error_matrix[i][j] = 频率 freq_vals[i], 噪声 noise_vals[j] 时的平均误差
    error_matrix = []
    total = len(freq_vals) * len(noise_vals)
    idx = 0

    for fi, freq in enumerate(freq_vals):
        row = []
        for nj, noise in enumerate(noise_vals):
            idx += 1
            if shutdown_flag.is_set():
                break
            rospy.loginfo("Trial %d/%d: freq=%.2f Hz, noise_std=%.4f m", idx, total, freq, noise)
            mean_err = run_single_trial(freq, noise, args.duration, recorder, workspace)
            if mean_err is None:
                row.append(float('nan'))
            else:
                row.append(mean_err)
                rospy.loginfo("  mean error = %.4f m", mean_err)
        if shutdown_flag.is_set():
            break
        error_matrix.append(row)

    # 若提前退出，补全矩阵为 nan 以便与 freq/noise 长度一致
    while len(error_matrix) < len(freq_vals):
        error_matrix.append([float('nan')] * len(noise_vals))
    for row in error_matrix:
        while len(row) < len(noise_vals):
            row.append(float('nan'))

    # 保存 JSON（转为 Python 原生类型以便序列化）
    def to_serializable(obj):
        if isinstance(obj, (list, np.ndarray)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if np.isfinite(v) else None
        return obj

    data = {
        'freq_Hz': to_serializable(freq_vals),
        'noise_std_m': to_serializable(noise_vals),
        'duration_sec': args.duration,
        'error_matrix': to_serializable(error_matrix),
        'timestamp': datetime.now().isoformat(),
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Saved %s", json_path)

    # 热力图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        M = np.array(error_matrix)
        if M.size > 0 and np.any(np.isfinite(M)):
            fig, ax = plt.subplots()
            im = ax.imshow(
                M,
                extent=[noise_vals[0], noise_vals[-1], freq_vals[0], freq_vals[-1]],
                aspect='auto',
                cmap='viridis',
                origin='lower',
            )
            ax.set_xlabel('noise_std', fontweight='bold')
            ax.set_ylabel('frequency', fontweight='bold')
            ax.set_title('Mean positioning error (m) vs freq & noise')
            cbar = plt.colorbar(im, ax=ax, label='error')
            cbar.set_label('error', fontweight='bold')
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
