#!/usr/bin/env python3
"""在线对比 4 种 dog_pos 方案的 XY 轨迹与滑窗误差。"""

import math
import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import rospy
from nav_msgs.msg import Odometry


METHODS = (
    ('ours', '/dog_pos_processed', 'b', 'Ours (cascade)'),
    ('kf', '/dog_pos_processed_kf', 'r', 'Whole KF'),
    ('lkf', '/dog_pos_processed_lkf', 'm', 'LKF incr'),
    ('vis', '/dog_pos_processed_vis', 'c', 'Vis fixed R/t'),
)


def interp_xy(buffer, t):
    """buffer: deque of (t, x, y). 线性插值，无数据返回 None。"""
    if not buffer:
        return None
    items = list(buffer)
    ts = np.array([e[0] for e in items])
    if t <= ts[0]:
        return items[0][1], items[0][2]
    if t >= ts[-1]:
        return items[-1][1], items[-1][2]
    i = int(np.searchsorted(ts, t))
    t0, x0, y0 = items[i - 1]
    t1, x1, y1 = items[i]
    if abs(t1 - t0) < 1e-9:
        return x0, y0
    a = (t - t0) / (t1 - t0)
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0)


def gt_window(gt_buf, t_now, window_sec):
    """取 [t_now-window_sec, t_now] 内 GT 样本；轨迹与误差共用。"""
    if not gt_buf:
        return []
    t_min = t_now - window_sec
    gt_win = [p for p in gt_buf if p[0] >= t_min]
    return gt_win if gt_win else [gt_buf[-1]]


def gt_speed_acc(gt_buf):
    """GT 当前 |v| 与 |a|（速度对时间差分）。"""
    if not gt_buf:
        return float('nan'), float('nan')
    t1, _, _, vx1, vy1, vz1 = gt_buf[-1]
    speed = math.sqrt(vx1 * vx1 + vy1 * vy1 + vz1 * vz1)
    if len(gt_buf) < 2:
        return speed, float('nan')
    t0, _, _, vx0, vy0, vz0 = gt_buf[-2]
    dt = t1 - t0
    if dt <= 1e-9:
        return speed, float('nan')
    ax = (vx1 - vx0) / dt
    ay = (vy1 - vy0) / dt
    az = (vz1 - vz0) / dt
    acc = math.sqrt(ax * ax + ay * ay + az * az)
    return speed, acc


def aligned_trail(gt_win, est_bufs):
    """以同一 GT 时间戳序列对齐 GT 与各路估计。"""
    gx, gy = [], []
    trails = {key: ([], []) for key, *_ in METHODS}
    for t, x, y, *_ in gt_win:
        gx.append(x)
        gy.append(y)
        for key, *_ in METHODS:
            est = interp_xy(est_bufs[key], t)
            xs, ys = trails[key]
            if est is None:
                xs.append(float('nan'))
                ys.append(float('nan'))
            else:
                xs.append(est[0])
                ys.append(est[1])
    return gx, gy, trails


def window_errors(gt_win, est_buf):
    """在同一 GT 时间戳上计算误差。"""
    errs = []
    for t, gx, gy, *_ in gt_win:
        est = interp_xy(est_buf, t)
        if est is None:
            continue
        ex, ey = est
        errs.append(math.hypot(gx - ex, gy - ey))
    if not errs:
        return float('nan'), float('nan'), 0
    arr = np.array(errs, dtype=float)
    return float(arr.mean()), float(np.sqrt(np.mean(arr * arr))), len(errs)


class DogPosCompareViz:
    def __init__(self):
        self.window_sec = float(rospy.get_param('~window_sec', 41.86))
        self.trail_sec = float(rospy.get_param('~trail_sec', 41.86))
        self.buffer_sec = float(rospy.get_param(
            '~buffer_sec', max(self.trail_sec, self.window_sec) + 5.0))
        self.refresh_hz = float(rospy.get_param('~refresh_hz', 10.0))

        self.gt_buf = deque()
        self.est_bufs = {key: deque() for key, *_ in METHODS}
        self.lock = threading.Lock()

        self.gt_sub = rospy.Subscriber(
            '/ground_truth_traj', Odometry, self._gt_cb, queue_size=50)
        self.est_subs = []
        for key, topic, *_ in METHODS:
            self.est_subs.append(rospy.Subscriber(
                topic, Odometry,
                lambda msg, k=key: self._est_cb(k, msg),
                queue_size=50))

        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.fig.subplots_adjust(bottom=0.28)
        self.line_gt, = self.ax.plot([], [], 'g-', linewidth=2.0, label='GT')
        self.lines = {}
        self.points = {}
        for key, _, color, label in METHODS:
            line, = self.ax.plot([], [], color + '-', linewidth=1.5, label=label)
            pt, = self.ax.plot([], [], color + 'o', markersize=6)
            self.lines[key] = line
            self.points[key] = pt
        self.pt_gt, = self.ax.plot([], [], 'go', markersize=7)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Dog Position Compare (XY, 4 methods)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.legend(loc='upper left', fontsize=8)
        self.stats_text = self.fig.text(
            0.08, 0.02, '',
            va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, _event):
        rospy.signal_shutdown('viz window closed')

    def _prune(self, buf, t_cut):
        while buf and buf[0][0] < t_cut:
            buf.popleft()

    def _push_gt(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        self.gt_buf.append((
            t,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            vx, vy, vz,
        ))
        self._prune(self.gt_buf, t - self.buffer_sec)

    def _push_est(self, buf, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()
        buf.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))
        self._prune(buf, t - self.buffer_sec)

    def _gt_cb(self, msg):
        with self.lock:
            self._push_gt(msg)

    def _est_cb(self, key, msg):
        with self.lock:
            self._push_est(self.est_bufs[key], msg)

    def _refresh(self, _frame):
        with self.lock:
            gt = list(self.gt_buf)
            est = {k: list(v) for k, v in self.est_bufs.items()}

        if not gt:
            return ()

        t_now = gt[-1][0]
        gt_trail = gt_window(gt, t_now, self.trail_sec)
        gt_win = gt_window(gt, t_now, self.window_sec)
        gx, gy, trails = aligned_trail(gt_trail, est)

        self.line_gt.set_data(gx, gy)
        self.pt_gt.set_data([gx[-1]], [gy[-1]])
        for key, _, _, _ in METHODS:
            px, py = trails[key]
            self.lines[key].set_data(px, py)
            self.points[key].set_data([px[-1]], [py[-1]])

        stat_lines = [
            f"Trail: {self.trail_sec:.1f} s ({len(gt_trail)} pts)  "
            f"Error window: {self.window_sec:.1f} s ({len(gt_win)} pts)",
        ]
        gt_speed, gt_acc = gt_speed_acc(gt)
        stat_lines.append(
            f"GT  |v|={gt_speed:.3f} m/s  |a|={gt_acc:.3f} m/s^2")

        for key, _, _, label in METHODS:
            mean_e, rmse_e, n_e = window_errors(gt_win, est[key])
            cur_e = float('nan')
            px, py = trails[key]
            if gx and not math.isnan(px[-1]):
                cur_e = math.hypot(gx[-1] - px[-1], gy[-1] - py[-1])
            stat_lines.append(
                f"{label:16s} now={cur_e:.3f} m  "
                f"mean={mean_e:.3f} m  rmse={rmse_e:.3f} m  (n={n_e})")

        self.stats_text.set_text('\n'.join(stat_lines))

        self.ax.relim()
        self.ax.autoscale_view()
        return ()

    def run(self):
        topics = ', '.join(f'{k}={t}' for k, t, *_ in METHODS)
        rospy.loginfo(
            "dog_pos_compare_viz: GT=/ground_truth_traj, %s, trail=%.1fs, window=%.1fs",
            topics, self.trail_sec, self.window_sec)

        spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        spin_thread.start()

        interval_ms = int(1000.0 / max(1.0, self.refresh_hz))
        self._ani = FuncAnimation(
            self.fig, self._refresh, interval=interval_ms,
            blit=False, cache_frame_data=False)
        plt.show()


def main():
    rospy.init_node('dog_pos_compare_viz')
    node = DogPosCompareViz()
    node.run()


if __name__ == '__main__':
    main()
