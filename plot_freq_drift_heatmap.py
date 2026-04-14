#!/usr/bin/env python3
"""
根据 experiment_freq_drift 生成的 results.json 重新绘制热力图。
带单位：frequency (Hz)、drift (rad/s, m/s)、error (m)；双线性插值平滑过渡。
用法：
  python3 plot_freq_drift_heatmap.py [--input results.json] [--output heatmap.png]
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='从 results.json 重绘 freq-drift 热力图（带单位、平滑）')
    parser.add_argument('--input', '-i', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'experiment_freq_drift_20260302_051951', 'results.json'),
                        help='输入的 results.json 路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出热力图路径，默认与 json 同目录的 heatmap.png')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    freq_vals = np.array(data['freq_Hz'])
    drift_vals = np.array(data['drift_rate'])
    M = np.array(data['error_matrix'])

    if M.size == 0:
        print('error_matrix 为空，无法绘图')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(
        M,
        extent=[drift_vals[0], drift_vals[-1], freq_vals[0], freq_vals[-1]],
        aspect='auto',
        cmap='viridis',
        origin='lower',
        interpolation='bilinear',
    )
    ax.set_xlabel('drift (rad/s, m/s)', fontweight='bold')
    ax.set_ylabel('frequency (Hz)', fontweight='bold')
    ax.set_title('Mean positioning error (m) vs target_ekf_odom freq & dog_pos drift')
    cbar = plt.colorbar(im, ax=ax, label='error (m)')
    cbar.set_label('error (m)', fontweight='bold')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')

    out_path = args.output
    if not out_path:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), 'heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved heatmap:', out_path)


if __name__ == '__main__':
    main()
