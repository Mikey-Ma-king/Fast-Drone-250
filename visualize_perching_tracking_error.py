#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot grouped bar chart of tracking error for different methods and trajectories.

This script reads a set of perching experiment bag files, computes the tracking
error between the UAV position and the target ground-truth position (from
`/target_ekf_odom`), and then plots
the mean error (with standard deviation as error bars) for:

  - Baseline
  - Frenet (FF)
  - Frenet + FF (ours)

over three trajectories (S, I, O).

Bag files (example names, adjust base_dir if needed):
  perching_2026-02-27-12-01-56_base_I.bag
  perching_2026-02-27-11-57-36_base_O.bag
  perching_2026-02-27-11-55-00_base_S.bag
  perching_2026-02-27-11-50-41_FF_S.bag
  perching_2026-02-27-11-45-39_FF_O.bag
  perching_2026-02-27-11-43-08_FF_I.bag
  perching_2026-02-27-11-35-33_ours_O.bag
  perching_2026-02-27-11-38-31_ours_I.bag
  perching_2026-02-27-12-05-04_ours_S.bag

Usage:
  cd /home/pc/Fast-Drone-250
  python3 plot_perching_tracking_error.py
"""

import os
import math
import argparse

import rosbag
import numpy as np

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend, only save to file
import matplotlib.pyplot as plt


def extract_positions(bag_path, topic):
    """
    Extract (time, position) from a nav_msgs/Odometry topic in a bag.

    Returns:
        times: np.ndarray, shape (N,)
        positions: np.ndarray, shape (N, 3)
    """
    times = []
    positions = []

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            pos = msg.pose.pose.position
            times.append(t.to_sec())
            positions.append([pos.x, pos.y, pos.z])

    if not times:
        return np.array([]), np.empty((0, 3))

    times = np.array(times)
    positions = np.array(positions)
    return times, positions


def compute_tracking_error(bag_path,
                           uav_topic="/vins_fusion/imu_propagate",
                           gt_topic="/target_ekf_odom",
                           max_time_diff=0.05,
                           min_time_from_start=0.0):
    """
    Compute tracking error norm between UAV and ground-truth trajectories.

    For each UAV sample, find the closest ground-truth sample in time (within
    max_time_diff) and compute the Euclidean distance between positions.

    Returns:
        errors: np.ndarray of shape (M,), where M is the number of matched samples.
    """
    uav_t, uav_p = extract_positions(bag_path, uav_topic)
    gt_t, gt_p = extract_positions(bag_path, gt_topic)

    if uav_t.size == 0 or gt_t.size == 0:
        print(f"[WARN] No data for '{uav_topic}' or '{gt_topic}' in {bag_path}")
        return np.array([])

    # reference start time: earliest timestamp across both topics
    t0 = min(uav_t[0], gt_t[0])

    errors = []
    j = 0
    for i in range(len(uav_t)):
        t = uav_t[i]

        # only use data after specified offset (seconds from start)
        if (t - t0) < min_time_from_start:
            continue

        # advance gt index while it is behind current time
        while j + 1 < len(gt_t) and gt_t[j + 1] <= t:
            j += 1

        # choose closer of gt_t[j] and gt_t[j+1] if available
        best_idx = j
        if j + 1 < len(gt_t):
            if abs(gt_t[j + 1] - t) < abs(gt_t[j] - t):
                best_idx = j + 1

        dt = abs(gt_t[best_idx] - t)
        if dt > max_time_diff:
            continue

        diff = uav_p[i] - gt_p[best_idx]
        err = math.sqrt(float(diff[0] ** 2 + diff[1] ** 2))
        errors.append(err)

    return np.array(errors)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracking error for perching experiments (grouped bar chart)."
    )
    parser.add_argument(
        "--bag_dir",
        type=str,
        default='/home/pc/perching_bag',
        help="Directory containing perching_*.bag files (default: current working directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/pc/Fast-Drone-250/perching_tracking_error_comparison.png",
        help="Output figure filename (relative to bag_dir if not absolute).",
    )
    parser.add_argument(
        "--t_after",
        type=float,
        default=5.0,
        help="Only use data after this many seconds from start of bag (default: 0.0).",
    )
    args = parser.parse_args()




    # Base directory where bag files are stored
    base_dir = args.bag_dir
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Bag directory does not exist: {base_dir}")
        return

    # Map (method, trajectory) -> bag filename
    # Methods: baseline, Frenet (FF), ours (Frenet+FF)
    # Trajectories: S, I, O
    bag_map = {
        ("Baseline", "Sine-curve"): "perching_2026-02-27-11-55-00_base_S.bag",
        ("Baseline", "Straight-line"): "perching_2026-02-27-12-01-56_base_I.bag",
        ("Baseline", "Circle"): "perching_2026-02-27-11-57-36_base_O.bag",
        ("Frenet", "Sine-curve"): "perching_2026-02-27-11-50-41_FF_S.bag",
        ("Frenet", "Straight-line"): "perching_2026-02-27-11-43-08_FF_I.bag",
        ("Frenet", "Circle"): "perching_2026-02-27-11-45-39_FF_O.bag",
        ("Ours", "Sine-curve"): "perching_2026-02-27-12-05-04_ours_S.bag",
        ("Ours", "Straight-line"): "perching_2026-02-27-11-38-31_ours_I.bag",
        ("Ours", "Circle"): "perching_2026-02-27-11-35-33_ours_O.bag",
    }

    methods = ["Baseline", "Frenet", "Ours"]
    # X 轴顺序：I, O, S
    trajectories = ["Straight-line", "Circle", "Sine-curve"]

    # Colors for each method (adjust as you like)
    colors = {
        "Baseline": "#1f77b4",  # blue
        "Frenet": "#ff7f0e",    # orange
        "Ours": "#2ca02c",      # green
    }

    # Collect statistics per (method, trajectory)
    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}
    errors_dict = {}  # (method, traj) -> np.ndarray of errors

    for traj in trajectories:
        for method in methods:
            fname = bag_map.get((method, traj))
            if fname is None:
                print(f"[WARN] No bag file defined for ({method}, {traj})")
                means[method].append(np.nan)
                stds[method].append(np.nan)
                errors_dict[(method, traj)] = np.array([])
                continue

            bag_path = os.path.join(base_dir, fname)
            if not os.path.isfile(bag_path):
                print(f"[WARN] Bag file not found: {bag_path}")
                means[method].append(np.nan)
                stds[method].append(np.nan)
                errors_dict[(method, traj)] = np.array([])
                continue

            print(f"[INFO] Processing {bag_path} ({method}, {traj}), t_after={args.t_after}s")
            errors = compute_tracking_error(bag_path, min_time_from_start=args.t_after)
            if errors.size == 0:
                means[method].append(np.nan)
                stds[method].append(np.nan)
                errors_dict[(method, traj)] = np.array([])
            else:
                means[method].append(errors.mean())
                stds[method].append(errors.std())
                errors_dict[(method, traj)] = errors

    # Prepare grouped "IQR bars": bar = middle 50% (Q1–Q3), whiskers = Q1-1.5IQR and Q3+1.5IQR
    x = np.arange(len(trajectories))  # 0,1,2 for S, I, O
    total_width = 0.8
    bar_width = total_width / len(methods)
    offset = -total_width / 2 + bar_width / 2

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for i, method in enumerate(methods):
        for j, traj in enumerate(trajectories):
            errs = errors_dict.get((method, traj), np.array([]))
            if errs.size == 0:
                continue

            q1 = np.percentile(errs, 25)
            q3 = np.percentile(errs, 75)
            median = np.median(errs)
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

            xpos = x[j] + offset + i * bar_width

            # Bar: middle 50% (Q1–Q3)
            ax.bar(
                xpos,
                q3 - q1,
                width=bar_width * 0.9,
                bottom=q1,
                color=colors.get(method, None),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
                label=method if j == 0 else None,  # one legend entry per method
            )

            # Whiskers
            ax.vlines(xpos, whisker_low, q1, colors="black", linewidth=1.0)
            ax.vlines(xpos, q3, whisker_high, colors="black", linewidth=1.0)
            cap_width = bar_width * 0.4
            ax.hlines(whisker_low, xpos - cap_width / 2, xpos + cap_width / 2, colors="black", linewidth=1.0)
            ax.hlines(whisker_high, xpos - cap_width / 2, xpos + cap_width / 2, colors="black", linewidth=1.0)

            # Median line inside the bar: extend across full bar width
            median_half_width = (bar_width * 0.9) / 2.0
            ax.hlines(median,
                      xpos - median_half_width,
                      xpos + median_half_width,
                      colors="black",
                      linewidth=1.2)

    # Success threshold line at 0.1 m
    success_threshold = 0.1
    ax.axhline(success_threshold, color="red", linestyle="--", linewidth=1.5)
    ax.text(
        0.02,
        success_threshold + 0.01,
        "Success Threshold (0.1 m)",
        color="red",
        fontsize=12,
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(trajectories, fontsize=12)
    ax.set_xlabel("Trajectory", fontsize=14)
    ax.set_ylabel("Tracking Error Norm (m)", fontsize=14)
    ax.set_title("Error Distribution Comparison", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)


    # Resolve output path
    if os.path.isabs(args.output):
        out_path = args.output
    else:
        out_path = os.path.join(base_dir, args.output)

    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved figure to {out_path}")


if __name__ == "__main__":
    main()

