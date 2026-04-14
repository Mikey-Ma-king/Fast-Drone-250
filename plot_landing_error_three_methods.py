#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比三种方法的着陆误差分布（类似 perching_tracking_error_comparison 的风格）。

数据目录：
  - experiment_landing_20260305_161842_vis  -> EO
  - experiment_landing_20260305_145210_lkf  -> IL
  - experiment_landing_20260305_143507_ours -> Ours

规则：只保留 error_xy <= 1.0 m 的样本，其余视为 outlier 丢弃后再统计和画图。
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


WORKSPACE = os.path.dirname(os.path.abspath(__file__))
RUNS = [
    ("EO",   os.path.join(WORKSPACE, "experiment_landing_20260305_161842_vis", "results.json")),
    ("IL",   os.path.join(WORKSPACE, "experiment_landing_20260305_145210_lkf", "results.json")),
    ("Ours", os.path.join(WORKSPACE, "experiment_landing_20260305_143507_ours", "results.json")),
]
MAX_ERR = 1.0  # m，超过这个阈值的 sample 直接丢弃


def load_filtered_errors(json_path, max_err=MAX_ERR):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    errs = []
    for exp in data.get("experiments", []):
        if "error_xy" in exp:
            v = float(exp["error_xy"])
            if v <= max_err:
                errs.append(v)
    return np.array(errs, dtype=float)


def main():
    labels = []
    all_errs = []
    for name, path in RUNS:
        if not os.path.isfile(path):
            print(f"[WARN] not found: {path}")
            continue
        errs = load_filtered_errors(path)
        print(f"[INFO] {name}: kept {errs.size} samples with error_xy <= {MAX_ERR} m")
        if errs.size == 0:
            print(f"[WARN] no valid error_xy after filtering in {path}")
            continue
        labels.append(name)
        all_errs.append(errs)

    if not all_errs:
        print("[ERROR] no valid data, exit.")
        return

    n = len(all_errs)
    x = np.arange(n)
    bar_width = 0.6

    plt.figure(figsize=(5.5, 4.2))
    ax = plt.gca()

    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # EO:red, IL:green, Ours:blue

    for i, errs in enumerate(all_errs):
        q1 = np.percentile(errs, 25)
        q3 = np.percentile(errs, 75)
        median = float(np.median(errs))
        iqr = q3 - q1

        if iqr == 0.0:
            whisker_low = q1
            whisker_high = q3
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            inlier_mask = (errs >= lower_bound) & (errs <= upper_bound)
            if np.any(inlier_mask):
                whisker_low = float(errs[inlier_mask].min())
                whisker_high = float(errs[inlier_mask].max())
            else:
                whisker_low = q1
                whisker_high = q3

        xpos = x[i]
        color = colors[i % len(colors)]

        # 中间 Q1–Q3 条形
        ax.bar(
            xpos,
            q3 - q1,
            width=bar_width,
            bottom=q1,
            color=color,
            alpha=0.8,
        )
        # 中位数
        ax.plot([xpos - bar_width * 0.5, xpos + bar_width * 0.5],
                [median, median],
                color="k",
                linewidth=2)
        # whisker
        ax.plot([xpos, xpos], [whisker_low, whisker_high], color="k", linewidth=1)
        ax.plot([xpos - bar_width * 0.25, xpos + bar_width * 0.25],
                [whisker_low, whisker_low], color="k", linewidth=1)
        ax.plot([xpos - bar_width * 0.25, xpos + bar_width * 0.25],
                [whisker_high, whisker_high], color="k", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=14)
    ax.set_ylabel("Landing error (m)", fontweight="bold", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.3)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=ymax * 1.05)

    out = os.path.join(WORKSPACE, "landing_error_three_methods_filtered.svg")
    plt.tight_layout()
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()

