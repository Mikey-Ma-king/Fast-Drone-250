#!/usr/bin/env python3
"""
将三个 freq-drift 实验结果画在一起：3 个子图，error 统一 vmax=0.3，frequency 5~30，drift 0~0.3。
vis 若只有 1 个 drift，则沿 drift 维复制成多列（与 drift 无关）。
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _rainbow_cmap():
    """彩虹色：蓝青绿黄橙红（低到高）"""
    return plt.cm.get_cmap('jet')

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
# 使用 2026-03-05 的三组新实验结果
DIRS = [
    ('(a) Ours', os.path.join(WORKSPACE, 'experiment_freq_drift_20260305_105042_ours')),
    ('(b) IL',   os.path.join(WORKSPACE, 'experiment_freq_drift_20260305_104020_lkf')),
    ('(c) EO',   os.path.join(WORKSPACE, 'experiment_freq_drift_20260305_111509_vis')),
]
FONTSIZE = 8
TITLE_FONTSIZE = 11
VMIN, VMAX = 0.0, 0.3
FREQ_RANGE = (5.0, 15.0)
DRIFT_RANGE = (0.0, 0.1)


def load_and_prepare(label, path):
    with open(os.path.join(path, 'results.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    freq = np.array(data['freq_Hz'])
    drift = np.array(data['drift_rate'])
    M = np.array(data['error_matrix'])
    # vis 可能只有 1 个 drift：沿 drift 复制，与 drift 无关
    if drift.size == 1:
        n_drift = 10  # 与常见实验一致
        drift = np.linspace(DRIFT_RANGE[0], DRIFT_RANGE[1], n_drift)
        M = np.tile(M, (1, n_drift))  # (n_freq, 1) -> (n_freq, n_drift)
    return freq, drift, M, label


def main():
    # 每个热力图正方形：总宽 = 3*边长 + colorbar，高 = 边长
    side = 2.0
    # 适当增加总宽度，但保持每个子图近似 side 的宽度
    # 原来宽度约 3*side+0.6=6.6，这里稍微放大到 7.0，并通过 wspace 留出空隙
    fig, axes = plt.subplots(1, 3, figsize=(9.0, side + 0.3), constrained_layout=False)
    for ax, (label, path) in zip(axes, DIRS):
        if not os.path.isdir(path):
            ax.set_title(f'{label} (no data)')
            continue
        freq, drift, M = load_and_prepare(label, path)[:3]
        im = ax.imshow(
            M,
            extent=[drift[0], drift[-1], freq[0], freq[-1]],
            aspect='auto',
            cmap=_rainbow_cmap(),
            origin='lower',
            interpolation='bilinear',
            vmin=VMIN,
            vmax=VMAX,
        )
        ax.set_xlim(DRIFT_RANGE)
        # 频率轴反过来：最大值在最下面
        ax.set_ylim(FREQ_RANGE[1], FREQ_RANGE[0])
        ax.set_xlabel('Drift magnitude', fontweight='bold', fontsize=FONTSIZE)
        ax.set_ylabel('Sensor frequency (Hz)', fontweight='bold', fontsize=FONTSIZE)
        ax.set_title(label, fontweight='bold', fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=FONTSIZE)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

    # 子图之间留一定间距，同时保证子图本身不被压缩太窄
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.4)

    cbar = plt.colorbar(im, ax=axes, label='Mean landing error (m)', shrink=1.0)
    cbar.set_label('Localization error (m)', fontweight='bold', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    out = os.path.join(WORKSPACE, 'experiment_freq_drift_three_compare.svg')
    plt.savefig(out, format='svg', bbox_inches='tight')
    plt.close()
    print('Saved:', out)


if __name__ == '__main__':
    main()
