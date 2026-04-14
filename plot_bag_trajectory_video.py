#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ROS bag 读取 ground truth 轨迹与 UAV 轨迹，在 2D xy 平面用不同颜色绘制，
并生成实时更新线条的动画视频。
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import rosbag
except ImportError:
    print("请先安装 rosbag：source /opt/ros/<distro>/setup.bash 或 pip install rosbag")
    raise


def extract_xy_from_bag(bag_path, topic, bag_start_time=None, start_time=None, end_time=None):
    """
    从 bag 中某一 topic 提取 (times, x, y)。
    支持 nav_msgs/Odometry 与 quadrotor_msgs/PositionCommand。

    Returns:
        times: (N,) 相对 bag 起始时间（秒）
        xs, ys: (N,) 位置 x, y
    """
    times = []
    xs = []
    ys = []

    with rosbag.Bag(bag_path, "r") as bag:
        info = bag.get_type_and_topic_info()
        if topic not in info[1]:
            return np.array([]), np.array([]), np.array([])
        first_ts = None
        for _, msg, t in bag.read_messages(topics=[topic]):
            if first_ts is None and bag_start_time is None:
                first_ts = t.to_sec()
            ts = t.to_sec()
            if bag_start_time is not None:
                rel = ts - bag_start_time
            else:
                rel = ts - first_ts if first_ts is not None else 0.0
            if start_time is not None and rel < start_time:
                continue
            if end_time is not None and rel > end_time:
                continue
            if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
                pos = msg.pose.pose.position
                x, y = pos.x, pos.y
            elif hasattr(msg, "position"):
                pos = msg.position
                x, y = pos.x, pos.y
            else:
                continue
            times.append(rel)
            xs.append(x)
            ys.append(y)

    if not times:
        return np.array([]), np.array([]), np.array([])
    return np.array(times), np.array(xs), np.array(ys)


def get_bag_start_time(bag_path):
    """获取 bag 第一条消息的时间戳（秒）。"""
    with rosbag.Bag(bag_path, "r") as bag:
        for _, _, t in bag.read_messages():
            return t.to_sec()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="从 bag 读取 GT 与 UAV 轨迹，生成 2D xy 实时更新线条视频"
    )
    parser.add_argument(
        "--bag",
        type=str,
        default="/home/pc/perching_bag/perching_2026-03-07-22-48-54.bag",
        help="bag 文件路径",
    )
    parser.add_argument(
        "--uav_topic",
        type=str,
        default="/vins_fusion/imu_propagate",
        help="UAV 轨迹 topic（Odometry/PositionCommand）",
    )
    parser.add_argument(
        "--gt_topic",
        type=str,
        default="/target_ekf_odom",
        help="Ground truth 轨迹 topic；默认先试 /ground_truth_traj，若无则用 /target_ekf_odom",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出视频路径；默认在 bag 同目录下，名为 <bag_basename>_traj_xy.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="从 bag 起始后多少秒开始（相对时间）",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="到 bag 起始后多少秒结束（相对时间）；默认用完整时长",
    )
    parser.add_argument(
        "--vis_after_sec",
        type=float,
        default=0.0,
        help="只可视化该秒数之后的轨迹（相对 bag 起始）；0 表示从开头画起",
    )
    parser.add_argument(
        "--gap_sec",
        type=float,
        default=0.5,
        help="若相邻两帧时间间隔超过该值（秒），则断开轨迹，从新一段开始画；0 表示不断开",
    )
    parser.add_argument(
        "--trail_sec",
        type=float,
        default=3.0,
        help="只显示当前时刻往前多少秒的轨迹，更早的删去（秒）",
    )
    parser.add_argument(
        "--uav_color",
        type=str,
        default="C0",
        help="UAV 轨迹颜色（matplotlib 颜色）",
    )
    parser.add_argument(
        "--gt_color",
        type=str,
        default="C1",
        help="Ground truth 轨迹颜色",
    )
    args = parser.parse_args()

    bag_path = os.path.abspath(args.bag)
    if not os.path.isfile(bag_path):
        print(f"错误：找不到 bag 文件 {bag_path}")
        return 1

    bag_start = get_bag_start_time(bag_path)
    if bag_start is None:
        print("错误：bag 为空或无法读取")
        return 1

    # 确定 GT topic
    with rosbag.Bag(bag_path, "r") as bag:
        topics = list(bag.get_type_and_topic_info()[1].keys())
    gt_topic = args.gt_topic
    if gt_topic is None:
        gt_topic = "/ground_truth_traj" if "/ground_truth_traj" in topics else "/target_ekf_odom"
    if gt_topic not in topics:
        print(f"警告：bag 中无 topic {gt_topic}，可选: {topics}")
        return 1
    if args.uav_topic not in topics:
        print(f"警告：bag 中无 UAV topic {args.uav_topic}")
        return 1

    print(f"UAV topic: {args.uav_topic}")
    print(f"GT  topic: {gt_topic}")

    uav_t, uav_x, uav_y = extract_xy_from_bag(
        bag_path, args.uav_topic, bag_start_time=bag_start,
        start_time=args.start, end_time=args.end
    )
    gt_t, gt_x, gt_y = extract_xy_from_bag(
        bag_path, gt_topic, bag_start_time=bag_start,
        start_time=args.start, end_time=args.end
    )

    if uav_t.size == 0:
        print("错误：UAV 轨迹无数据")
        return 1
    if gt_t.size == 0:
        print("错误：Ground truth 轨迹无数据")
        return 1

    # 统一时间范围用于动画；若指定只可视化 N 秒后，则动画从该时刻开始
    t_min = min(uav_t.min(), gt_t.min())
    t_max = max(uav_t.max(), gt_t.max())
    vis_after = max(t_min, float(args.vis_after_sec))
    t_min_vis = vis_after
    duration = t_max - t_min_vis
    if duration <= 0:
        duration = 1.0

    num_frames = max(60, int(args.fps * duration))
    time_per_frame = duration / num_frames

    # 超过 gap_sec 无数据则断开轨迹：标记每段结束位置（下一帧与当前帧间隔 > gap_sec）
    gap_sec = max(0.0, float(args.gap_sec))
    uav_break_after = np.zeros(len(uav_t), dtype=bool)  # uav_break_after[i]=True 表示 i 与 i+1 之间断开
    if len(uav_t) > 1 and gap_sec > 0:
        uav_break_after[:-1] = np.diff(uav_t) > gap_sec
    gt_break_after = np.zeros(len(gt_t), dtype=bool)
    if len(gt_t) > 1 and gap_sec > 0:
        gt_break_after[:-1] = np.diff(gt_t) > gap_sec

    # 输出路径
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        bag_dir = os.path.dirname(bag_path)
        bag_basename = os.path.splitext(os.path.basename(bag_path))[0]
        out_path = os.path.join(bag_dir, f"{bag_basename}_traj_xy.mp4")

    # 图形与动画：整体小一点，中间图小一点，字大一点
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_position([0.14, 0.14, 0.82, 0.82])  # 中间图略小，四周留白
    ax.set_xlabel("x (m)", fontsize=16)
    ax.set_ylabel("y (m)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 用全轨迹范围固定 xy 坐标轴（若只可视化 N 秒后，仍用全部数据算范围，保证比例一致）
    x_margin = max(0.5, (gt_x.max() - gt_x.min() + uav_x.max() - uav_x.min()) / 2 * 0.1)
    y_margin = max(0.5, (gt_y.max() - gt_y.min() + uav_y.max() - uav_y.min()) / 2 * 0.1)
    ax.set_xlim(min(gt_x.min(), uav_x.min()) - x_margin, max(gt_x.max(), uav_x.max()) + x_margin)
    ax.set_ylim(min(gt_y.min(), uav_y.min()) - y_margin, max(gt_y.max(), uav_y.max()) + y_margin)

    # 用于动画的线（实时更新，逐帧变长）
    line_gt, = ax.plot([], [], color=args.gt_color, linewidth=2, label="Ground truth")
    line_uav, = ax.plot([], [], color=args.uav_color, linewidth=2, label="UAV")
    ax.legend(loc="upper left", fontsize=14)
    title_text = ax.set_title("", fontsize=16)

    trail_sec = max(0.0, float(args.trail_sec))
    t_min_vis_trail = lambda t_cur: max(vis_after, t_cur - trail_sec)

    def build_segmented(t_arr, x_arr, y_arr, break_after, t_cur):
        """在 [t_cur-trail_sec, t_cur] 且 >= vis_after 的点中，在断点处插入 NaN，使轨迹断开。"""
        t_lo = t_min_vis_trail(t_cur)
        x_list, y_list = [], []
        for i in range(len(t_arr)):
            if not (t_lo <= t_arr[i] <= t_cur):
                continue
            if x_list and i > 0 and break_after[i - 1]:
                x_list.append(np.nan)
                y_list.append(np.nan)
            x_list.append(x_arr[i])
            y_list.append(y_arr[i])
        return (np.array(x_list), np.array(y_list)) if x_list else (np.array([]), np.array([]))

    def init():
        line_gt.set_data([], [])
        line_uav.set_data([], [])
        title_text.set_text("")
        return line_gt, line_uav, title_text

    def animate(frame):
        t_cur = t_min_vis + (frame / max(1, num_frames - 1)) * duration
        x_gt, y_gt = build_segmented(gt_t, gt_x, gt_y, gt_break_after, t_cur)
        x_uav, y_uav = build_segmented(uav_t, uav_x, uav_y, uav_break_after, t_cur)
        line_gt.set_data(x_gt, y_gt)
        line_uav.set_data(x_uav, y_uav)
        title_text.set_text(f"t = {t_cur:.2f} s")
        return line_gt, line_uav, title_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=num_frames,
        interval=1000.0 / args.fps, blit=True, repeat=True
    )

    # 写入视频（优先 ffmpeg）
    try:
        Writer = animation.FFMpegWriter(fps=args.fps)
    except Exception as e1:
        try:
            Writer = animation.PillowWriter(fps=args.fps)
        except Exception as e2:
            print(f"无法创建视频写入器 (ffmpeg: {e1}, pillow: {e2})")
            return 1

    if vis_after > t_min:
        print(f"只可视化 t >= {vis_after:.1f} s 的轨迹")
    print(f"正在写入视频: {out_path} ({num_frames} 帧, {args.fps} fps)")
    anim.save(out_path, writer=Writer, dpi=100)
    plt.close()
    print("完成。")
    return 0


if __name__ == "__main__":
    exit(main())
