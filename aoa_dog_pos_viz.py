#!/usr/bin/env python3
"""在线可视化 /dog_pos_aoa_debug 与 /dog_pos_processed 的 XY 轨迹。"""

import math
import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rospy
from nav_msgs.msg import Odometry


TOPICS = (
    ('aoa', '/dog_pos_aoa_debug', '#d62728', 'AOA debug'),
    ('proc', '/dog_pos_processed', '#1f77b4', 'Dog pos processed'),
)


class AoaDogPosViz:
    def __init__(self):
        self.trail_sec = float(rospy.get_param('~trail_sec', 30.0))
        self.buffer_sec = float(rospy.get_param(
            '~buffer_sec', self.trail_sec + 5.0))
        self.refresh_hz = float(rospy.get_param('~refresh_hz', 15.0))
        self.equal_aspect = bool(rospy.get_param('~equal_aspect', True))

        self.bufs = {key: deque() for key, *_ in TOPICS}
        self.lock = threading.Lock()
        self.subs = []
        for key, topic, *_ in TOPICS:
            self.subs.append(rospy.Subscriber(
                topic, Odometry,
                lambda msg, k=key: self._cb(k, msg),
                queue_size=50))

        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.fig.subplots_adjust(bottom=0.18)
        self.lines = {}
        self.points = {}
        for key, _, color, label in TOPICS:
            line, = self.ax.plot(
                [], [], '-', color=color, linewidth=1.8, label=label, alpha=0.85)
            pt, = self.ax.plot(
                [], [], 'o', color=color, markersize=8, markeredgecolor='white',
                markeredgewidth=1.0)
            self.lines[key] = line
            self.points[key] = pt

        # 当前两点连线，方便看偏差
        self.link_line, = self.ax.plot(
            [], [], '--', color='#7f7f7f', linewidth=1.0, alpha=0.7,
            label='AOA ↔ processed')

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('AOA debug vs dog_pos_processed (XY)')
        self.ax.grid(True, alpha=0.3)
        if self.equal_aspect:
            self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.legend(loc='upper left', fontsize=9)

        self.stats_text = self.fig.text(
            0.08, 0.02, '',
            va='bottom', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, _event):
        rospy.signal_shutdown('viz window closed')

    def _prune(self, buf, t_cut):
        while buf and buf[0][0] < t_cut:
            buf.popleft()

    def _cb(self, key, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()
        with self.lock:
            self.bufs[key].append((
                t,
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ))
            self._prune(self.bufs[key], t - self.buffer_sec)

    @staticmethod
    def _trail(buf, t_now, trail_sec):
        t_min = t_now - trail_sec
        win = [p for p in buf if p[0] >= t_min]
        return win if win else (list(buf[-1:]) if buf else [])

    @staticmethod
    def _hz(buf, window=2.0):
        if len(buf) < 2:
            return 0.0
        t_end = buf[-1][0]
        n = sum(1 for p in buf if p[0] >= t_end - window)
        return n / window if window > 0 else 0.0

    def _refresh(self, _frame):
        with self.lock:
            snaps = {k: list(v) for k, v in self.bufs.items()}

        t_now = 0.0
        for buf in snaps.values():
            if buf:
                t_now = max(t_now, buf[-1][0])
        if t_now <= 0.0:
            self.stats_text.set_text('waiting for /dog_pos_aoa_debug and /dog_pos_processed ...')
            return ()

        trails = {}
        for key, *_ in TOPICS:
            trails[key] = self._trail(snaps[key], t_now, self.trail_sec)

        for key, *_ in TOPICS:
            tr = trails[key]
            xs = [p[1] for p in tr]
            ys = [p[2] for p in tr]
            self.lines[key].set_data(xs, ys)
            if xs:
                self.points[key].set_data([xs[-1]], [ys[-1]])
            else:
                self.points[key].set_data([], [])

        aoa = trails['aoa']
        proc = trails['proc']
        if aoa and proc:
            ax_, ay_ = aoa[-1][1], aoa[-1][2]
            px, py = proc[-1][1], proc[-1][2]
            self.link_line.set_data([ax_, px], [ay_, py])
            dx = ax_ - px
            dy = ay_ - py
            dz = aoa[-1][3] - proc[-1][3]
            err_xy = math.hypot(dx, dy)
            err_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            age = abs(aoa[-1][0] - proc[-1][0])
        else:
            self.link_line.set_data([], [])
            err_xy = err_3d = age = float('nan')

        lines = [
            f"trail={self.trail_sec:.1f}s   "
            f"AOA Hz={self._hz(snaps['aoa']):5.1f}   "
            f"processed Hz={self._hz(snaps['proc']):5.1f}",
        ]
        if aoa:
            lines.append(
                f"AOA       N={len(aoa):4d}  "
                f"x={aoa[-1][1]:7.3f}  y={aoa[-1][2]:7.3f}  z={aoa[-1][3]:7.3f}")
        else:
            lines.append('AOA       (no data)')
        if proc:
            lines.append(
                f"processed N={len(proc):4d}  "
                f"x={proc[-1][1]:7.3f}  y={proc[-1][2]:7.3f}  z={proc[-1][3]:7.3f}")
        else:
            lines.append('processed (no data)')
        if not math.isnan(err_xy):
            lines.append(
                f"diff XY={err_xy:.3f} m   3D={err_3d:.3f} m   "
                f"|t_aoa-t_proc|={age*1000:.0f} ms")
        self.stats_text.set_text('\n'.join(lines))

        self.ax.relim()
        self.ax.autoscale_view()
        return ()

    def run(self):
        rospy.loginfo(
            "aoa_dog_pos_viz: topics=%s, trail=%.1fs, refresh=%.1fHz",
            ', '.join(t for _, t, *_ in TOPICS),
            self.trail_sec, self.refresh_hz)

        spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        spin_thread.start()

        interval_ms = int(1000.0 / max(1.0, self.refresh_hz))
        self._ani = FuncAnimation(
            self.fig, self._refresh, interval=interval_ms,
            blit=False, cache_frame_data=False)
        plt.show()


def main():
    rospy.init_node('aoa_dog_pos_viz')
    AoaDogPosViz().run()


if __name__ == '__main__':
    main()
