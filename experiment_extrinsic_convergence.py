#!/usr/bin/env python3
"""
外参实验（四方案 × 两轨迹 × n 重复）：

实验 A — jump_response：初始 R/t 跳变，漂移=0
  指标：IAE（收敛前 ∫|e|dt）、收敛时间 t_conv

实验 B — drift：无初始跳变，持续漂移
  指标：稳定段 mean / max / min 误差（ours/kf 的 mean/max 从收敛后起算）
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

METHODS = (
    ('ours', '/dog_pos_processed', 'Ours (cascade)', 'b'),
    ('kf', '/dog_pos_processed_kf', 'Whole KF', 'r'),
    ('lkf', '/dog_pos_processed_lkf', 'LKF incr', 'm'),
    ('vis', '/dog_pos_processed_vis', 'Vis fixed R/t', 'c'),
)

TRAJ_MODES = {
    'wavy_circle': '/target_wavy_circle_publisher',
    'sin_accel_straight': '/target_sin_accel_straight_publisher',
}

# 与 fake_target EXTRINSIC_DRIFT_DEFAULTS 一致
DEFAULT_DRIFT = {
    'dog_pos_drift_rate_x': 0.008,
    'dog_pos_drift_rate_y': 0.008,
    'dog_pos_drift_rate_z': 0.0,
    'dog_pos_drift_yaw_rate': 0.003,
    'vins_drift_rate_x': -0.008,
    'vins_drift_rate_y': -0.008,
    'vins_drift_rate_z': 0.0,
    'vins_drift_yaw_rate': -0.003,
}

CONVERGENCE_ERR_M = 0.15
CONVERGENCE_SUSTAIN_SEC = 0.5
INITIAL_WINDOW_SEC = 5.0
DRIFT_WARMUP_SEC = 3.0
# Exp B drift：mean/max 从收敛后起算的方案
POST_CONV_DRIFT_METHODS = ('ours', 'kf')

EXTRINSIC_PARAM_KEYS = (
    'dog_pos_drift_offset_x', 'dog_pos_drift_offset_y', 'dog_pos_drift_offset_z',
    'dog_pos_drift_rate_x', 'dog_pos_drift_rate_y', 'dog_pos_drift_rate_z',
    'dog_pos_drift_yaw_initial', 'dog_pos_drift_yaw_rate',
    'vins_drift_rate_x', 'vins_drift_rate_y', 'vins_drift_rate_z', 'vins_drift_yaw_rate',
)

active_processes = {'launch': None, 'fake_target': None}
active_processes_lock = threading.Lock()
shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    rospy.logwarn('Interrupt received, shutting down...')
    shutdown_flag.set()
    cleanup_processes()


def cleanup_processes():
    with active_processes_lock:
        for name in ('fake_target', 'launch'):
            proc = active_processes.get(name)
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception as exc:
                rospy.logwarn('Error stopping %s: %s', name, exc)
            active_processes[name] = None


def clear_fake_target_params(ns):
    for key in EXTRINSIC_PARAM_KEYS:
        full = f'{ns}/{key}'
        try:
            if rospy.has_param(full):
                rospy.delete_param(full)
        except Exception:
            pass


def set_jump_response_params(ns, yaw_rad, offset_m):
    """实验 A：仅初始跳变，漂移速率全 0。"""
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', float(yaw_rad))
    rospy.set_param(f'{ns}/dog_pos_drift_offset_x', float(offset_m))
    rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_rate_x', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_rate_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_rate_z', 0.0)
    rospy.set_param(f'{ns}/vins_drift_rate_x', 0.0)
    rospy.set_param(f'{ns}/vins_drift_rate_y', 0.0)
    rospy.set_param(f'{ns}/vins_drift_rate_z', 0.0)
    rospy.set_param(f'{ns}/vins_drift_yaw_rate', 0.0)


def set_drift_params(ns, drift_cfg):
    """实验 B：无初始跳变，启用持续漂移。"""
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', float(drift_cfg['dog_pos_drift_yaw_rate']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_x', float(drift_cfg['dog_pos_drift_rate_x']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_y', float(drift_cfg['dog_pos_drift_rate_y']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_z', float(drift_cfg['dog_pos_drift_rate_z']))
    rospy.set_param(f'{ns}/vins_drift_rate_x', float(drift_cfg['vins_drift_rate_x']))
    rospy.set_param(f'{ns}/vins_drift_rate_y', float(drift_cfg['vins_drift_rate_y']))
    rospy.set_param(f'{ns}/vins_drift_rate_z', float(drift_cfg['vins_drift_rate_z']))
    rospy.set_param(f'{ns}/vins_drift_yaw_rate', float(drift_cfg['vins_drift_yaw_rate']))


class MultiMethodRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.t0 = None
        self.est_bufs = {key: deque(maxlen=8000) for key, *_ in METHODS}
        self.series = {key: [] for key, *_ in METHODS}
        self.ready_times = {key: None for key, *_ in METHODS}

        self._gt_sub = rospy.Subscriber(
            '/ground_truth_traj', Odometry, self._gt_cb, queue_size=100)
        for key, topic, *_ in METHODS:
            rospy.Subscriber(
                topic, Odometry,
                lambda msg, k=key: self._est_cb(k, msg),
                queue_size=100)

    def reset(self):
        with self.lock:
            self.t0 = None
            for key in self.est_bufs:
                self.est_bufs[key].clear()
                self.series[key] = []
                self.ready_times[key] = None

    def _est_cb(self, key, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()
        pos_ready = msg.pose.pose.orientation.w >= 0.5
        yaw_ready = msg.pose.pose.orientation.x >= 0.5
        with self.lock:
            if key == 'ours' and pos_ready and yaw_ready and self.ready_times[key] is None:
                if self.t0 is not None:
                    self.ready_times[key] = t - self.t0
            self.est_bufs[key].append((
                t, msg.pose.pose.position.x,
                msg.pose.pose.position.y, msg.pose.pose.position.z,
            ))

    @staticmethod
    def _interp(buf, t):
        if not buf:
            return None
        items = sorted(buf, key=lambda e: e[0])
        ts = np.array([e[0] for e in items])
        if t <= ts[0]:
            e = items[0]
            return e[1], e[2], e[3]
        if t >= ts[-1]:
            e = items[-1]
            return e[1], e[2], e[3]
        i = int(np.searchsorted(ts, t))
        t0, x0, y0, z0 = items[i - 1]
        t1, x1, y1, z1 = items[i]
        if abs(t1 - t0) < 1e-9:
            return x0, y0, z0
        a = (t - t0) / (t1 - t0)
        return x0 + a * (x1 - x0), y0 + a * (y1 - y0), z0 + a * (z1 - z0)

    def _gt_cb(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y
        gz = msg.pose.pose.position.z
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            t_rel = t - self.t0
            bufs = {k: list(v) for k, v in self.est_bufs.items()}
        for key, *_ in METHODS:
            est = self._interp(bufs[key], t)
            if est is None:
                continue
            err = math.sqrt(
                (gx - est[0]) ** 2 + (gy - est[1]) ** 2 + (gz - est[2]) ** 2)
            with self.lock:
                self.series[key].append((t_rel, err))

    def get_jump_metrics(self, conv_err=CONVERGENCE_ERR_M, sustain=CONVERGENCE_SUSTAIN_SEC):
        with self.lock:
            out_series = {k: list(v) for k, v in self.series.items()}
            ready_times = dict(self.ready_times)
        metrics = {}
        for key, *_ in METHODS:
            pts = out_series.get(key, [])
            if not pts:
                metrics[key] = {
                    'iae_pre_conv': float('nan'),
                    't_conv_err_s': None,
                    't_conv_ready_s': ready_times.get(key),
                    'n_samples': 0,
                }
                continue
            ts = np.array([p[0] for p in pts], dtype=float)
            errs = np.array([p[1] for p in pts], dtype=float)
            t_conv = _detect_convergence(ts, errs, conv_err, sustain)
            iae = _compute_iae_pre_conv(ts, errs, t_conv)
            metrics[key] = {
                'iae_pre_conv': iae,
                't_conv_err_s': None if not math.isfinite(t_conv) else float(t_conv),
                't_conv_ready_s': ready_times.get(key),
                'n_samples': int(len(errs)),
            }
        return out_series, metrics

    def get_drift_metrics(self, warmup_sec=DRIFT_WARMUP_SEC,
                          conv_err=CONVERGENCE_ERR_M, sustain=CONVERGENCE_SUSTAIN_SEC):
        with self.lock:
            out_series = {k: list(v) for k, v in self.series.items()}
            ready_times = dict(self.ready_times)
        metrics = {}
        for key, *_ in METHODS:
            pts = out_series.get(key, [])
            if not pts:
                metrics[key] = {
                    'mean_error_m': float('nan'),
                    'max_error_m': float('nan'),
                    'min_error_m': float('nan'),
                    't_conv_err_s': None,
                    't_conv_ready_s': ready_times.get(key),
                    'n_samples': 0,
                }
                continue
            ts = np.array([p[0] for p in pts], dtype=float)
            errs = np.array([p[1] for p in pts], dtype=float)
            mask_w = ts >= warmup_sec
            use_w = errs[mask_w] if np.any(mask_w) else errs

            if key in POST_CONV_DRIFT_METHODS:
                t_conv = _detect_convergence(ts, errs, conv_err, sustain)
                t_ready = ready_times.get(key) if key == 'ours' else None
                post_mean, post_max = _compute_post_conv_stats(ts, errs, t_conv, t_ready)
                t_start = _post_conv_start_time(t_conv, t_ready if key == 'ours' else None)
                n_post = int(np.sum(ts >= t_start)) if t_start is not None else 0
                metrics[key] = {
                    'mean_error_m': post_mean,
                    'max_error_m': post_max,
                    'min_error_m': float(np.min(use_w)),
                    't_conv_err_s': None if not math.isfinite(t_conv) else float(t_conv),
                    't_conv_ready_s': t_ready,
                    'n_samples': n_post if n_post > 0 else int(len(use_w)),
                }
            else:
                metrics[key] = {
                    'mean_error_m': float(np.mean(use_w)),
                    'max_error_m': float(np.max(use_w)),
                    'min_error_m': float(np.min(use_w)),
                    'n_samples': int(len(use_w)),
                }
        return out_series, metrics


def _post_conv_start_time(t_conv, t_ready=None):
    if t_conv is None or not math.isfinite(t_conv):
        return None
    t_start = float(t_conv)
    if t_ready is not None and math.isfinite(t_ready):
        t_start = max(t_start, float(t_ready))
    return t_start


def _compute_post_conv_stats(ts, errs, t_conv, t_ready=None):
    """收敛后段 mean / max；起点为 max(t_conv, t_ready)（t_ready 仅 ours 使用）。"""
    t_start = _post_conv_start_time(t_conv, t_ready)
    if t_start is None or len(errs) == 0:
        return float('nan'), float('nan')
    mask = ts >= t_start
    if not np.any(mask):
        return float('nan'), float('nan')
    e_use = errs[mask]
    return float(np.mean(e_use)), float(np.max(e_use))


def _metrics_from_drift_series(series, conv_err=CONVERGENCE_ERR_M,
                               sustain=CONVERGENCE_SUSTAIN_SEC,
                               warmup_sec=DRIFT_WARMUP_SEC, ready_times=None):
    """由单次 run 的误差序列重算 Exp B 指标（plot-only / 旧 results 兼容）。"""
    ready_times = ready_times or {}
    metrics = {}
    for key, *_ in METHODS:
        pts = series.get(key, [])
        if not pts:
            metrics[key] = {
                'mean_error_m': float('nan'),
                'max_error_m': float('nan'),
                'min_error_m': float('nan'),
                't_conv_err_s': None,
                't_conv_ready_s': ready_times.get(key),
                'n_samples': 0,
            }
            continue
        ts = np.array([p[0] for p in pts], dtype=float)
        errs = np.array([p[1] for p in pts], dtype=float)
        mask_w = ts >= warmup_sec
        use_w = errs[mask_w] if np.any(mask_w) else errs
        t_ready = ready_times.get(key)

        if key in POST_CONV_DRIFT_METHODS:
            t_conv = _detect_convergence(ts, errs, conv_err, sustain)
            post_mean, post_max = _compute_post_conv_stats(
                ts, errs, t_conv, t_ready if key == 'ours' else None)
            t_start = _post_conv_start_time(t_conv, t_ready if key == 'ours' else None)
            n_post = int(np.sum(ts >= t_start)) if t_start is not None else 0
            metrics[key] = {
                'mean_error_m': post_mean,
                'max_error_m': post_max,
                'min_error_m': float(np.min(use_w)),
                't_conv_err_s': None if not math.isfinite(t_conv) else float(t_conv),
                't_conv_ready_s': t_ready,
                'n_samples': n_post if n_post > 0 else int(len(use_w)),
            }
        else:
            metrics[key] = {
                'mean_error_m': float(np.mean(use_w)),
                'max_error_m': float(np.max(use_w)),
                'min_error_m': float(np.min(use_w)),
                'n_samples': int(len(use_w)),
            }
    return metrics


def _compute_iae_pre_conv(ts, errs, t_conv):
    """收敛前 IAE：∫|e(t)| dt，积分区间 [0, t_conv]；未收敛则积到全程。"""
    if len(ts) < 2:
        return float('nan')
    if t_conv is not None and math.isfinite(t_conv) and t_conv > 0.0:
        mask = ts <= t_conv
    else:
        mask = np.ones(len(ts), dtype=bool)
    t_use = ts[mask]
    e_use = errs[mask]
    if len(t_use) < 2:
        return float('nan')
    return float(np.trapz(e_use, t_use))


def _stat_from_values(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        'n': len(vals),
    }


def _detect_convergence(ts, errs, threshold, sustain_sec):
    if len(ts) < 5:
        return float('nan')
    t_end = ts[-1]
    t = max(1.0, ts[0])
    while t <= t_end - sustain_sec:
        mask = (ts >= t) & (ts <= t + sustain_sec)
        if np.sum(mask) >= 3 and float(np.mean(errs[mask])) < threshold:
            return float(t)
        t += 0.05
    return float('nan')


def _compute_phase_means(ts, errs, t_conv, initial_window_sec=INITIAL_WINDOW_SEC):
    if len(errs) == 0:
        return float('nan'), float('nan')
    pre_end = initial_window_sec
    if math.isfinite(t_conv) and t_conv > 0.05:
        pre_end = min(pre_end, t_conv)
    pre_mask = ts <= pre_end
    pre_mean = float(np.mean(errs[pre_mask])) if np.any(pre_mask) else float('nan')
    if math.isfinite(t_conv) and t_conv > 0.05:
        post_mask = ts >= t_conv
        post_mean = float(np.mean(errs[post_mask])) if np.any(post_mask) else float('nan')
    else:
        post_mean = float('nan')
    return pre_mean, post_mean


def run_single_trial(mode, duration_sec, recorder, workspace, experiment_type,
                     yaw_init, offset_m, drift_cfg, conv_err):
    if shutdown_flag.is_set():
        return None, None

    ns = TRAJ_MODES[mode]
    clear_fake_target_params(ns)
    time.sleep(0.2)
    if experiment_type == 'jump_response':
        set_jump_response_params(ns, yaw_init, offset_m)
    else:
        set_drift_params(ns, drift_cfg)
    recorder.reset()

    env = os.environ.copy()
    env['FAKE_TARGET_MODE'] = mode

    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=3)
            except Exception:
                pass
        active_processes['fake_target'] = subprocess.Popen(
            [sys.executable, FAKE_TARGET_SCRIPT],
            cwd=workspace, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    time.sleep(1.5)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None, None

    cmd = f'source "{SETUP_BASH}" && roslaunch planning dog_pos_processor_compare.launch'
    with active_processes_lock:
        if active_processes['launch'] is not None:
            try:
                active_processes['launch'].terminate()
                active_processes['launch'].wait(timeout=3)
            except Exception:
                pass
        active_processes['launch'] = subprocess.Popen(
            ['bash', '-c', cmd], cwd=workspace,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    time.sleep(3.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None, None

    recorder.reset()
    elapsed = 0.0
    while elapsed < duration_sec and not shutdown_flag.is_set():
        time.sleep(0.1)
        elapsed += 0.1

    if experiment_type == 'jump_response':
        series, metrics = recorder.get_jump_metrics(conv_err=conv_err)
    else:
        series, metrics = recorder.get_drift_metrics(conv_err=conv_err)

    cleanup_processes()
    time.sleep(1.0)
    return series, metrics


def plot_error_curves(exp_name, exp_title, traj_results, out_dir, conv_err=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    for mode, data in traj_results.items():
        series = data['series']
        metrics = data['metrics']
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key, _, label, color in METHODS:
            pts = series.get(key, [])
            if not pts:
                continue
            ts = [p[0] for p in pts]
            errs = [p[1] for p in pts]
            ax.plot(ts, errs, color=color, linewidth=1.2, label=label, alpha=0.9)
            if exp_name == 'jump_response':
                t_conv = metrics[key].get('t_conv_err_s')
                if t_conv is not None and math.isfinite(t_conv):
                    ax.axvline(t_conv, color=color, linestyle='--', linewidth=0.8, alpha=0.45)

        if conv_err is not None and exp_name == 'jump_response':
            ax.axhline(conv_err, color='gray', linestyle=':', linewidth=0.8,
                       label=f'conv thr={conv_err:.2f} m')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position error vs GT (m)')
        ax.set_title(f'{exp_title} — {mode}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        path = os.path.join(exp_dir, f'error_vs_time_{mode}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        rospy.loginfo('Saved %s', path)


def plot_summary_bars(exp_name, exp_title, traj_results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [m[2] for m in METHODS]
    keys = [m[0] for m in METHODS]
    modes = list(traj_results.keys())
    if not modes:
        return

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), squeeze=False)
    for ax_idx, mode in enumerate(modes):
        ax = axes[0, ax_idx]
        metrics = traj_results[mode]['metrics']
        if exp_name == 'jump_response':
            vals = [metrics[k].get('iae_pre_conv', float('nan')) for k in keys]
            ylabel = 'IAE pre-convergence (m·s)'
            title_suffix = 'IAE'
        else:
            vals = [metrics[k].get('mean_error_m', float('nan')) for k in keys]
            ylabel = 'Mean error (m)'
            title_suffix = 'mean'
        vals = [v if v is not None and math.isfinite(v) else 0.0 for v in vals]
        x = np.arange(len(keys))
        ax.bar(x, vals, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{mode}\n({title_suffix})')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(exp_title, fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, exp_name, 'metrics_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    rospy.loginfo('Saved %s', path)


def aggregate_repeats(raw_by_traj, exp_name):
    """对多次 trial 的 metrics 取均值/标准差。"""
    jump_metrics = ('iae_pre_conv', 't_conv_err_s')
    drift_metrics = ('mean_error_m', 'max_error_m', 'min_error_m')
    metric_names = jump_metrics if exp_name == 'jump_response' else drift_metrics

    agg = {}
    for mode, run_list in raw_by_traj.items():
        agg[mode] = {}
        for key, _, label, _ in METHODS:
            agg[mode][key] = {'label': label, 'n_total': len(run_list)}
            for mname in metric_names:
                vals = []
                for metrics in run_list:
                    v = metrics[key].get(mname)
                    if v is not None and math.isfinite(v):
                        vals.append(float(v))
                agg[mode][key][mname] = _stat_from_values(vals)
            if exp_name == 'jump_response':
                nc = agg[mode][key]['t_conv_err_s']['n']
                agg[mode][key]['n_conv'] = nc
    return agg


def _fmt_cell(stat, fmt='.2f', nc_note=False, n_total=5):
    m, s = stat.get('mean', float('nan')), stat.get('std', float('nan'))
    n = stat.get('n', 0)
    if m is None or not math.isfinite(m):
        if nc_note and n_total:
            return f'NC (0/{n_total})'
        return '—'
    cell = f'{m:{fmt}}±{s:{fmt}}' if math.isfinite(s) and s > 0 else f'{m:{fmt}}'
    if nc_note and n < n_total:
        cell += f' ({n}/{n_total})'
    return cell


def save_summary_table(all_agg, out_dir, repeats):
    """CSV + Markdown：Exp A (IAE, t_conv)，Exp B (mean, max, min)。"""
    rows = []
    method_keys = [m[0] for m in METHODS]
    method_labels = {m[0]: m[2] for m in METHODS}
    traj_list = ('wavy_circle', 'sin_accel_straight')

    lines = [f'# Extrinsic experiment summary (n={repeats} repeats per cell)', '']

    def add_table(title, exp_key, metric_key, fmt, nc_note=False):
        lines.extend(['', f'## {title}', '| Method | wavy_circle | sin_accel_straight |',
                      '|--------|-------------|-------------------|'])
        for key in method_keys:
            cells = []
            for mode in traj_list:
                a = all_agg.get(exp_key, {}).get(mode, {}).get(key, {})
                stat = a.get(metric_key, {})
                nt = a.get('n_total', repeats)
                cells.append(_fmt_cell(stat, fmt=fmt, nc_note=nc_note, n_total=nt))
                rows.append({
                    'experiment': exp_key,
                    'trajectory': mode,
                    'method': key,
                    'method_label': method_labels[key],
                    'metric': metric_key,
                    'mean': stat.get('mean'),
                    'std': stat.get('std'),
                    'n_valid': stat.get('n'),
                    'n_repeats': nt,
                })
            lines.append(f'| {method_labels[key]} | {cells[0]} | {cells[1]} |')

    add_table('Exp A: Jump — IAE pre-convergence (m·s)', 'jump_response', 'iae_pre_conv', '.2f')
    add_table('Exp A: Jump — convergence time (s)', 'jump_response', 't_conv_err_s', '.2f', nc_note=True)
    add_table('Exp B: Drift — mean error (m, ours/kf post-conv)', 'drift', 'mean_error_m', '.3f')
    add_table('Exp B: Drift — max error (m, ours/kf post-conv)', 'drift', 'max_error_m', '.3f')
    add_table('Exp B: Drift — min error (m)', 'drift', 'min_error_m', '.3f')

    csv_path = os.path.join(out_dir, 'summary_table.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('experiment,trajectory,method,method_label,metric,mean,std,n_valid,n_repeats\n')
        for r in rows:
            mean_v = '' if r['mean'] is None or not math.isfinite(r['mean']) else r['mean']
            std_v = '' if r['std'] is None or not math.isfinite(r['std']) else r['std']
            f.write(
                f"{r['experiment']},{r['trajectory']},{r['method']},{r['method_label']},"
                f"{r['metric']},{mean_v},{std_v},{r.get('n_valid','')},{r.get('n_repeats','')}\n")

    md_path = os.path.join(out_dir, 'summary_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    rospy.loginfo('Saved %s', csv_path)
    rospy.loginfo('Saved %s', md_path)
    return lines


def plot_unified_summary(all_agg, out_dir, repeats):
    """统一图 5×2：A(IAE,t_conv) + B(mean,max,min) × 两轨迹。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    method_keys = [m[0] for m in METHODS]
    method_labels = [m[2] for m in METHODS]
    colors = [m[3] for m in METHODS]
    traj_modes = ['wavy_circle', 'sin_accel_straight']

    panels = [
        ('jump_response', 'iae_pre_conv', 'Exp A: IAE pre-conv (m·s)', False),
        ('jump_response', 't_conv_err_s', 'Exp A: Convergence time (s)', True),
        ('drift', 'mean_error_m', 'Exp B: Mean error (m, ours/kf post-conv)', False),
        ('drift', 'max_error_m', 'Exp B: Max error (m, ours/kf post-conv)', False),
        ('drift', 'min_error_m', 'Exp B: Min error (m)', False),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 3.2 * len(panels)))
    x = np.arange(len(method_keys))
    width = 0.55

    for row, (exp_key, metric_key, ylabel, nc_mode) in enumerate(panels):
        for col, mode in enumerate(traj_modes):
            ax = axes[row, col]
            means, stds, valid = [], [], []
            for key in method_keys:
                stat = all_agg.get(exp_key, {}).get(mode, {}).get(key, {}).get(metric_key, {})
                m = stat.get('mean', float('nan'))
                means.append(m if m is not None else float('nan'))
                stds.append(stat.get('std', 0.0))
                valid.append(m is not None and math.isfinite(m))
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)
            bar_colors = [colors[i] if valid[i] else '#cccccc' for i in range(len(method_keys))]
            ax.bar(x, np.where(valid, means, 0), width,
                   yerr=np.where(valid, stds, 0), capsize=3,
                   color=bar_colors, alpha=0.88, edgecolor='white')
            if nc_mode:
                for i in range(len(method_keys)):
                    if not valid[i]:
                        ax.text(i, 0.5, 'NC', ha='center', va='bottom', fontsize=7, color='gray')
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=22, ha='right', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(mode, fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Extrinsic experiments (mean ± std, n={repeats})', fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, 'summary_unified.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    rospy.loginfo('Saved %s', path)


def plot_averaged_error_curves(exp_name, raw_by_traj, out_dir, conv_err=None):
    """多次 run 的误差曲线按时间对齐后取均值（可选）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    grid = np.linspace(0, 55, 551)

    for mode, run_list in raw_by_traj.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key, _, label, color in METHODS:
            all_interp = []
            for run in run_list:
                pts = run.get('series', {}).get(key, [])
                if not pts:
                    continue
                ts = np.array([p[0] for p in pts])
                errs = np.array([p[1] for p in pts])
                if len(ts) < 2:
                    continue
                all_interp.append(np.interp(grid, ts, errs, left=np.nan, right=np.nan))
            if not all_interp:
                continue
            arr = np.array(all_interp)
            mean_e = np.nanmean(arr, axis=0)
            std_e = np.nanstd(arr, axis=0)
            ax.plot(grid, mean_e, color=color, linewidth=1.4, label=label)
            ax.fill_between(grid, mean_e - std_e, mean_e + std_e,
                            color=color, alpha=0.15)

        if conv_err is not None and exp_name == 'jump_response':
            ax.axhline(conv_err, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position error vs GT (m)')
        ax.set_title(f'{exp_name} — {mode} (mean ± std over repeats)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        path = os.path.join(exp_dir, f'error_mean_{mode}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        try:
            rospy.loginfo('Saved %s', path)
        except Exception:
            print('Saved', path)


def regenerate_plots_from_json(json_path, conv_err=CONVERGENCE_ERR_M):
    """从已保存 results.json 重新生成曲线子目录与汇总图（无需 roscore）。"""
    out_dir = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repeats = data.get('repeats', 5)
    jump_meta = data.get('experiments', {}).get('jump_response', {})
    drift_meta = data.get('experiments', {}).get('drift', {})
    conv_thr = jump_meta.get('convergence_threshold_m',
                             drift_meta.get('convergence_threshold_m', conv_err))
    warmup_sec = drift_meta.get('warmup_sec', DRIFT_WARMUP_SEC)

    all_agg = {}
    for exp_name, exp_meta in data.get('experiments', {}).items():
        raw_by_traj = {}
        for mode, traj_data in exp_meta.get('trajectories', {}).items():
            run_records = []
            for run in traj_data.get('runs', []):
                if 'metrics' not in run:
                    continue
                if exp_name == 'drift' and run.get('series'):
                    ready_times = {
                        key: run['metrics'].get(key, {}).get('t_conv_ready_s')
                        for key, *_ in METHODS
                    }
                    run['metrics'] = _metrics_from_drift_series(
                        run['series'],
                        conv_err=conv_thr,
                        warmup_sec=warmup_sec,
                        ready_times=ready_times,
                    )
                run_records.append(run)
            raw_by_traj[mode] = [r['metrics'] for r in run_records if 'metrics' in r]
        if raw_by_traj:
            all_agg[exp_name] = aggregate_repeats(raw_by_traj, exp_name)

    for exp_name, exp_meta in data.get('experiments', {}).items():
        runs_by_mode = {
            m: traj_data['runs']
            for m, traj_data in exp_meta.get('trajectories', {}).items()
            if traj_data.get('runs')
        }
        if runs_by_mode:
            plot_averaged_error_curves(
                exp_name, runs_by_mode, out_dir,
                conv_err=conv_err if exp_name == 'jump_response' else None)

    if all_agg:
        save_summary_table(all_agg, out_dir, repeats)
        plot_unified_summary(all_agg, out_dir, repeats)
    print(f'Regenerated plots in {out_dir}')


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def log_metrics(exp_name, mode, metrics):
    for key, _, label, _ in METHODS:
        m = metrics[key]
        if exp_name == 'jump_response':
            t_conv = m.get('t_conv_err_s')
            rospy.loginfo(
                '  [%s/%s] %s: IAE=%.2f m·s, t_conv=%s s',
                exp_name, mode, label,
                m.get('iae_pre_conv', float('nan')),
                f'{t_conv:.2f}' if t_conv is not None else 'nan',
            )
        else:
            extra = ''
            if key in POST_CONV_DRIFT_METHODS:
                t_conv = m.get('t_conv_err_s')
                extra = f', t_conv={t_conv:.2f}s' if t_conv is not None else ', t_conv=nan'
            rospy.loginfo(
                '  [%s/%s] %s: mean=%.3f max=%.3f min=%.3f m%s',
                exp_name, mode, label,
                m['mean_error_m'], m['max_error_m'], m['min_error_m'],
                extra,
            )


def main():
    parser = argparse.ArgumentParser(description='外参实验：跳变响应 + 漂移（四方案 × 两轨迹）')
    parser.add_argument('--duration', type=float, default=55.0)
    parser.add_argument('--repeats', type=int, default=5, help='每种配置重复次数')
    parser.add_argument('--yaw-init-deg', type=float, default=90.0)
    parser.add_argument('--offset-m', type=float, default=0.5)
    parser.add_argument('--conv-err', type=float, default=CONVERGENCE_ERR_M)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--modes', nargs='+', default=['wavy_circle', 'sin_accel_straight'])
    parser.add_argument('--experiments', nargs='+',
                        default=['jump_response', 'drift'],
                        choices=['jump_response', 'drift'])
    parser.add_argument('--plot-curves', action='store_true',
                        help='额外保存每次 run 的单次误差曲线 error_vs_time_*.png')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='仅从已有 results.json 重新生成曲线与汇总图，不跑实验')
    args = parser.parse_args()

    if args.plot_only:
        regenerate_plots_from_json(args.plot_only, args.conv_err)
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rospy.init_node('experiment_extrinsic_batch', anonymous=True)

    if not os.path.isfile(SETUP_BASH):
        rospy.logerr('Missing %s', SETUP_BASH)
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(
        WORKSPACE, f'experiment_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    yaw_rad = math.radians(args.yaw_init_deg)
    recorder = MultiMethodRecorder()
    time.sleep(0.5)

    all_output = {
        'timestamp': datetime.now().isoformat(),
        'duration_sec': args.duration,
        'repeats': args.repeats,
        'experiments': {},
    }
    all_agg = {}

    exp_titles = {
        'jump_response': 'Exp A: Jump response (drift rate=0)',
        'drift': 'Exp B: Continuous drift (no initial jump)',
    }

    for exp_name in args.experiments:
        if shutdown_flag.is_set():
            break
        rospy.loginfo('======== %s (×%d repeats) ========', exp_titles[exp_name], args.repeats)
        raw_by_traj = {mode: [] for mode in args.modes if mode in TRAJ_MODES}
        exp_meta = {
            'title': exp_titles[exp_name],
            'repeats': args.repeats,
            'trajectories': {},
        }
        if exp_name == 'jump_response':
            exp_meta['initial_yaw_deg'] = args.yaw_init_deg
            exp_meta['initial_offset_m'] = args.offset_m
            exp_meta['drift_rates'] = 'all zero'
            exp_meta['convergence_threshold_m'] = args.conv_err
        else:
            exp_meta['initial_jump'] = 'none'
            exp_meta['drift_rates'] = DEFAULT_DRIFT
            exp_meta['warmup_sec'] = DRIFT_WARMUP_SEC
            exp_meta['convergence_threshold_m'] = args.conv_err
            exp_meta['post_conv_mean_max_methods'] = list(POST_CONV_DRIFT_METHODS)

        for mode in args.modes:
            if mode not in TRAJ_MODES:
                continue
            run_records = []
            for rep in range(args.repeats):
                if shutdown_flag.is_set():
                    break
                rospy.loginfo('--- %s / %s  repeat %d/%d ---',
                                exp_name, mode, rep + 1, args.repeats)
                series, metrics = run_single_trial(
                    mode, args.duration, recorder, WORKSPACE,
                    exp_name, yaw_rad, args.offset_m, DEFAULT_DRIFT, args.conv_err)
                if series is None:
                    continue
                run_records.append({'series': series, 'metrics': metrics})
                log_metrics(exp_name, mode, metrics)

            raw_by_traj[mode] = [r['metrics'] for r in run_records]
            if args.plot_curves and run_records:
                traj_results = {mode: run_records[-1]}
                plot_error_curves(
                    exp_name, exp_titles[exp_name], traj_results, out_dir,
                    conv_err=args.conv_err if exp_name == 'jump_response' else None)

            exp_meta['trajectories'][mode] = {
                'runs': [
                    {'metrics': r['metrics'],
                     'series': {k: [[float(a), float(b)] for a, b in r['series'].get(k, [])]
                                for k in r['series']}}
                    for r in run_records
                ],
            }

        all_agg[exp_name] = aggregate_repeats(
            {m: raw_by_traj[m] for m in raw_by_traj if raw_by_traj[m]}, exp_name)
        exp_meta['aggregated'] = all_agg[exp_name]
        all_output['experiments'][exp_name] = exp_meta

        # 默认：两实验均生成 error_mean_*.png 到 jump_response/ / drift/
        if exp_meta['trajectories']:
            plot_averaged_error_curves(
                exp_name,
                {m: exp_meta['trajectories'][m]['runs'] for m in exp_meta['trajectories']},
                out_dir,
                conv_err=args.conv_err if exp_name == 'jump_response' else None)

    cleanup_processes()

    all_output['aggregated'] = all_agg
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(all_output), f, indent=2, ensure_ascii=False)
    rospy.loginfo('Saved %s', json_path)

    if all_agg:
        table_lines = save_summary_table(all_agg, out_dir, args.repeats)
        plot_unified_summary(all_agg, out_dir, args.repeats)
        rospy.loginfo('--- Summary ---')
        for line in table_lines:
            if line.startswith('|') and not line.startswith('|--'):
                rospy.loginfo(line)

    rospy.loginfo('Done. Results in %s', out_dir)


if __name__ == '__main__':
    main()
