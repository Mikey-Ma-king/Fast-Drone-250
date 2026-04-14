#!/usr/bin/env python3
"""绘制「丢失后平均误差 vs 漂移」对比图：Ours、LKF 两条曲线，vis 用叉号标注 target lost。"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR_OURS = '/home/pc/Fast-Drone-250/experiment_drift_error_20260305_131041_ours'
DIR_LKF = '/home/pc/Fast-Drone-250/experiment_drift_error_20260305_132546_lkf'
OUT_PATH = '/home/pc/Fast-Drone-250/plot_drift_loss_compare.svg'


def load_results(dir_path):
    p = os.path.join(dir_path, 'results.json')
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return np.array(d['drift_rate']), np.array(d['mean_error_m'])


def main():
    drift_ours, err_ours = load_results(DIR_OURS)
    drift_lkf, err_lkf = load_results(DIR_LKF)

    fig, ax = plt.subplots()
    ax.plot(drift_ours, err_ours, 'b-o', linewidth=2, markersize=6, label='Ours')
    ax.plot(drift_lkf, err_lkf, 'g-s', linewidth=2, markersize=6, label='IL')

    # EO: 在 y=1.5 处画横线，NAN 像 1.2、1.4 一样标在 y 轴外侧
    y_nan = 1.5
    ax.axhline(y=y_nan, color='red', linestyle='-', linewidth=2, label='EO (target lost)')

    ax.set_xlabel('Drift rate (rad/s or m/s)', fontweight='bold')
    ax.set_ylabel('Mean positioning error (m)', fontweight='bold')
    ax.set_title('Mean error after loss vs drift')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # 保证 y 轴包含 1.5，并在 1.5 处显示刻度 "NAN"（与 1.2、1.4 一样在坐标轴外侧）
    ax.set_ylim(top=max(ax.get_ylim()[1], y_nan + 0.1))
    locs = list(ax.get_yticks())
    if y_nan not in locs:
        locs.append(y_nan)
        locs.sort()
        ax.set_yticks(locs)
    labels = [str(round(t, 1)) if abs(t - y_nan) > 0.01 else 'NAN' for t in ax.get_yticks()]
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig(OUT_PATH, format='svg', bbox_inches='tight')
    plt.close()
    print('Saved', OUT_PATH)


if __name__ == '__main__':
    main()
