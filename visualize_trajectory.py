#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and visualize 3D trajectories from ROS bag files
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse

def extract_trajectory_from_bag(bag_path, topics=None, start_time=None, end_time=None):
    """
    Extract trajectory data from bag file
    
    Args:
        bag_path: Path to bag file
        topics: List of topics to extract, if None auto-detect
        start_time: Start time in seconds (relative to bag start), if None use bag start
        end_time: End time in seconds (relative to bag start), if None use bag end
    
    Returns:
        dict: Dictionary containing trajectory data for each topic
    """
    if not os.path.exists(bag_path):
        print(f"Error: Bag file does not exist: {bag_path}")
        return None
    
    print(f"Reading bag file: {bag_path}")
    
    # Default topic list
    if topics is None:
        topics = [
            '/vins_fusion/imu_propagate',  # Drone actual position
            '/position_cmd',                # Position command
            '/target_ekf_odom',            # Target position
            '/dog_pos_processed',           # Dog position
            '/ground_truth_traj'           # Ground truth target position 
        ]
    
    trajectories = {}
    bag_start_time = None
    
    try:
        bag = rosbag.Bag(bag_path, 'r')
        
        # Get all topics in bag
        bag_topics = bag.get_type_and_topic_info()[1].keys()
        print(f"\nTopics in bag file:")
        for topic in bag_topics:
            print(f"  - {topic}")
        
        # Get bag start time from first message
        for topic_name, msg, t in bag.read_messages():
            bag_start_time = t.to_sec()
            break
        
        if bag_start_time is None:
            print("Error: Bag file is empty")
            bag.close()
            return None
        
        print(f"\nBag start time: {bag_start_time:.2f} s")
        if start_time is not None:
            print(f"Filter start time: {start_time:.2f} s (absolute: {bag_start_time + start_time:.2f} s)")
        if end_time is not None:
            print(f"Filter end time: {end_time:.2f} s (absolute: {bag_start_time + end_time:.2f} s)")
        
        # Extract data for each topic
        for topic in topics:
            if topic not in bag_topics:
                print(f"Warning: Topic {topic} not in bag file, skipping")
                continue
            
            print(f"\nExtracting topic: {topic}")
            positions = []
            times = []
            
            for topic_name, msg, t in bag.read_messages(topics=[topic]):
                try:
                    # Filter by time range
                    relative_time = t.to_sec() - bag_start_time
                    if start_time is not None and relative_time < start_time:
                        continue
                    if end_time is not None and relative_time > end_time:
                        continue
                    
                    # Extract position based on message type
                    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                        # Odometry message
                        pos = msg.pose.pose.position
                        positions.append([pos.x, pos.y, pos.z])
                        times.append(relative_time)
                    elif hasattr(msg, 'position'):
                        # PositionCommand message
                        pos = msg.position
                        positions.append([pos.x, pos.y, pos.z])
                        times.append(relative_time)
                    else:
                        print(f"Warning: Cannot parse message type for topic {topic}")
                        continue
                except Exception as e:
                    print(f"Warning: Error processing message: {e}")
                    continue
            
            if len(positions) > 0:
                trajectories[topic] = {
                    'positions': np.array(positions),
                    'times': np.array(times)
                }
                print(f"  Extracted {len(positions)} position points")
            else:
                print(f"  Topic {topic} has no valid data")
        
        bag.close()
        print(f"\nSuccessfully extracted {len(trajectories)} trajectories")
        
    except Exception as e:
        print(f"Error: Failed to read bag file: {e}")
        return None
    
    return trajectories

def visualize_trajectories(trajectories, output_path=None, selected_topics=None,
                          xlim=None, ylim=None, zlim=None, title=None, show_legend=True):
    """
    Visualize 3D trajectories

    Args:
        trajectories: Dictionary of trajectory data
        output_path: Path to save image, if None display image
        selected_topics: List of topics to visualize, if None visualize all
        xlim, ylim, zlim: Optional (min, max) for each axis; if None, auto from data
        title: Figure title; if None, use default
        show_legend: Whether to show the legend
    """
    if not trajectories:
        print("Error: No trajectory data to visualize")
        return
    
    # Filter trajectories based on selection
    if selected_topics is not None:
        trajectories = {topic: data for topic, data in trajectories.items() 
                       if topic in selected_topics}
        if not trajectories:
            print("Error: No valid trajectories selected")
            return
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors and labels（包括 ground_truth_traj）
    colors = {
        '/vins_fusion/imu_propagate': 'blue',
        '/position_cmd': 'red',
        '/target_ekf_odom': 'green',
        '/dog_pos_processed': 'orange',
        '/ground_truth_traj': 'black',
    }
    
    labels = {
        '/vins_fusion/imu_propagate': 'Drone Actual Trajectory',
        '/position_cmd': 'Position Command Trajectory',
        '/target_ekf_odom': 'Target estimation',
        '/dog_pos_processed': 'Dog Position Trajectory',
        '/ground_truth_traj': 'Ground Truth Target Trajectory',
    }
    
    # Collect all end points for calculating average landing point
    end_points = []
    
    # UAV 轨迹延后 2s 再开始画（只对 /vins_fusion/imu_propagate 生效）
    uav_delay_s = 5
    positions_for_limits = []  # 仅用实际绘制的点算坐标上下限

    for topic, data in trajectories.items():
        positions = data['positions']
        times = data['times']
        if len(positions) == 0:
            continue

        # UAV：只画 t >= t_start + 2s 的部分
        if topic == '/vins_fusion/imu_propagate' and len(times) > 0:
            t0 = times[0]
            mask = times >= t0 + uav_delay_s
            if not np.any(mask):
                continue
            positions = positions[mask]
            times = times[mask]

        positions_for_limits.append(positions)
        color = colors.get(topic, 'gray')
        label = labels.get(topic, topic)

        # Plot trajectory line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color=color, label=label, linewidth=2, alpha=0.7)

        # Mark starting point with red circle（延后后的起点）
        if len(positions) > 0:
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      color='red', marker='o', s=150, zorder=10)
            end_points.append(positions[-1])
    
    # Calculate and plot average landing point with red star
    if len(end_points) > 0:
        end_points_array = np.array(end_points)
        avg_landing_point = np.mean(end_points_array, axis=0)
        ax.scatter(avg_landing_point[0], avg_landing_point[1], avg_landing_point[2],
                  color='red', marker='*', s=500, zorder=10, label='Landing Point')
    
    # Add starting point label to legend (only once)
    ax.scatter([], [], [], color='red', marker='o', s=150, label='Starting Point')
    
    # Set axis labels 与刻度数字；轴标签 ZYX 比刻度数字大一点
    ax.set_xlabel('X (m)', fontsize=18, labelpad=16)
    ax.set_ylabel('Y (m)', fontsize=18, labelpad=16)
    ax.set_zlabel('Z (m)', fontsize=18, labelpad=24)   # Z 标签需要更多空间，避免裁到图外
    ax.tick_params(axis='x', labelsize=16, pad=4)   # xy 数字往里一点
    ax.tick_params(axis='y', labelsize=16, pad=4)
    ax.tick_params(axis='z', labelsize=16, pad=14)
    if show_legend:
        ax.legend(loc='best', fontsize=16)
    
    # x/y/z 上下限：若手动指定则用指定值，否则按数据自动加边距
    limits_spec = (xlim, ylim, zlim)
    all_positions = np.vstack(positions_for_limits) if len(positions_for_limits) > 0 else None
    margin_ratio, min_margin = 0.05, 0.05
    for i, name in enumerate(['x', 'y', 'z']):
        if limits_spec[i] is not None:
            lo, hi = limits_spec[i][0], limits_spec[i][1]
        elif all_positions is not None and len(all_positions) > 0:
            lo, hi = all_positions[:, i].min(), all_positions[:, i].max()
            r = hi - lo
            margin = max(r * margin_ratio, min_margin) if r > 0 else min_margin
            lo, hi = lo - margin, hi + margin
        else:
            continue
        getattr(ax, f'set_{name}lim')(lo, hi)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # 3D 子图不受 subplots_adjust/tight_layout 影响，用 set_position 缩小绘图区、右侧多留白给 Z 轴
    plt.tight_layout(pad=0.5)
    # [left, bottom, width, height] 单位 figure 比例 0~1；width 小一点让右侧有足够空间给 Z(m)
    ax.set_position([0.05, 0.04, 0.58, 0.92])
    
    # Save or display（矢量图用 format=pdf/svg，无需 dpi）；pad_inches 加大避免裁边
    if output_path:
        fmt = (os.path.splitext(output_path)[1] or '.pdf').lstrip('.').lower()
        if fmt == 'pdf' or fmt == 'svg':
            plt.savefig(output_path, format=fmt, bbox_inches='tight', pad_inches=0.35)
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.35)
        print(f"\nSaved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Extract and visualize 3D trajectories from ROS bag files')
    parser.add_argument('--bag', type=str, 
                       default='/home/pc/perching_bag/perching_2026-02-12-05-41-15.bag',
                       help='Path to bag file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (if not specified, auto-generate from bag filename)')
    parser.add_argument('--start_time', type=float, default=None,
                       help='Start time in seconds (relative to bag start)')
    parser.add_argument('--end_time', type=float, default=None,
                       help='End time in seconds (relative to bag start)')
    parser.add_argument('--topics', type=str, nargs='+', default=None,
                       help='Topics to visualize (e.g., /vins_fusion/imu_propagate /position_cmd). If not specified, visualize all available topics')
    parser.add_argument('--format', '-f', type=str, choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format: pdf/svg 矢量图, png 位图 (default: pdf)')
    parser.add_argument('--xlim', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                       help='X 轴范围 (m)，如 --xlim 0 10')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                       help='Y 轴范围 (m)')
    parser.add_argument('--zlim', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                       help='Z 轴范围 (m)')
    parser.add_argument('--title', type=str, default=None,
                       help='图标题，如 "Stairs: QDR state estimation and UAV landing trajectory"')
    parser.add_argument('--no-legend', action='store_true',
                       help='不显示图例')
    args = parser.parse_args()

    args.topics = ['/vins_fusion/imu_propagate', '/target_ekf_odom',]
    # args.topics = ['/vins_fusion/imu_propagate', '/ground_truth_traj']
    
    # args.bag = '/home/pc/perching_bag/perching_2026-03-04-14-33-28.bag'
    # args.start_time = 435.0
    # args.end_time = 452.5
    # args.ylim = [-2, 2]
    # args.title = 'Stairs: QDR state estimation and UAV landing trajectory'

    args.bag = '/home/pc/perching_bag/perching_2026-02-27-15-10-27.bag'
    args.start_time = 265.0
    args.end_time = 281.0
    args.ylim = [-2, 2]
    # args.title = 'Lawn: QDR state estimation and UAV landing trajectory'


    # args.bag = '/home/pc/perching_bag/perching_2026-02-27-15-03-58.bag'
    # args.start_time = 830.0
    # args.end_time = 850.0
    # args.ylim = [-4, 4]
    # args.title = 'Corridor: QDR state estimation and UAV landing trajectory'

    # args.bag = '/home/pc/Fast-Drone-250/experiment_landing_20260305_143507_ours'
    # args.bag = '/home/pc/Fast-Drone-250/experiment_landing_20260305_161842_vis'
    # args.bag = '/home/pc/Fast-Drone-250/experiment_landing_20260305_145210_lkf'

    args.format = 'svg'
    args.no_legend = True

    # Default output directory is Fast-Drone-250 folder
    default_output_dir = '/home/pc/Fast-Drone-250'
    ext = '.' + args.format

    # 支持：若 --bag/args.bag 是目录：
    #   - 若该目录下直接有 .bag，则逐个可视化，输出到 default_output_dir
    #   - 若该目录下是若干子目录，则在每个子目录里寻找 .bag，并把输出图保存在对应子目录中
    if os.path.isdir(args.bag):
        bag_dir = args.bag
        entries = sorted(os.listdir(bag_dir))
        direct_bags = [
            os.path.join(bag_dir, f)
            for f in entries
            if f.endswith('.bag') and os.path.isfile(os.path.join(bag_dir, f))
        ]
        subdirs = [
            os.path.join(bag_dir, d)
            for d in entries
            if os.path.isdir(os.path.join(bag_dir, d))
        ]

        targets = []
        # 1) 目录下直接的 bag
        for bag_path in direct_bags:
            targets.append((bag_path, default_output_dir))
        # 2) 子目录中的 bag（输出放在子目录中）
        for sub in subdirs:
            for f in sorted(os.listdir(sub)):
                if f.endswith('.bag') and os.path.isfile(os.path.join(sub, f)):
                    bag_path = os.path.join(sub, f)
                    targets.append((bag_path, sub))

        if not targets:
            print(f"No .bag files found in directory or its subdirectories: {bag_dir}")

        for bag_path, out_dir in targets:
            bag_basename = os.path.splitext(os.path.basename(bag_path))[0]

            # 生成输出文件名
            if args.output is None:
                output_path = os.path.join(
                    out_dir,
                    f'trajectory_visualization_{bag_basename}{ext}',
                )
            else:
                output_dir = out_dir
                output_basename = os.path.basename(args.output)
                output_name, output_ext = os.path.splitext(output_basename)
                if output_ext == '':
                    output_ext = ext
                output_path = os.path.join(output_dir, f'{output_name}_{bag_basename}{output_ext}')

            print(f"\n[INFO] Processing bag: {bag_path}")
            trajectories = extract_trajectory_from_bag(
                bag_path,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            if not trajectories:
                print("  Failed to extract trajectory data")
                continue

            xlim = tuple(args.xlim) if args.xlim is not None else None
            ylim = tuple(args.ylim) if args.ylim is not None else None
            zlim = tuple(args.zlim) if args.zlim is not None else None
            visualize_trajectories(
                trajectories,
                output_path,
                selected_topics=args.topics,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                title=args.title,
                show_legend=not args.no_legend,
            )

            # 打印统计信息
            print("\n=== Trajectory Statistics ===")
            for topic, data in trajectories.items():
                positions = data['positions']
                if len(positions) > 0:
                    print(f"\n{topic}:")
                    print(f"  Points: {len(positions)}")
                    print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
                    print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
                    print(f"  Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
                    if len(data['times']) > 0:
                        duration = data['times'][-1] - data['times'][0]
                        print(f"  Duration: {duration:.2f} s")
                        print(f"  Time range: [{data['times'][0]:.2f}, {data['times'][-1]:.2f}] s")
    else:
        # 单个 bag 文件的旧逻辑
        bag_basename = os.path.splitext(os.path.basename(args.bag))[0]
        if args.output is None:
            output_path = os.path.join(
                default_output_dir,
                f'trajectory_visualization_{bag_basename}{ext}',
            )
        else:
            output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else default_output_dir
            output_basename = os.path.basename(args.output)
            output_name, output_ext = os.path.splitext(output_basename)
            if output_ext == '':
                output_ext = ext
            output_path = os.path.join(output_dir, f'{output_name}_{bag_basename}{output_ext}')

        trajectories = extract_trajectory_from_bag(
            args.bag,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        if trajectories:
            xlim = tuple(args.xlim) if args.xlim is not None else None
            ylim = tuple(args.ylim) if args.ylim is not None else None
            zlim = tuple(args.zlim) if args.zlim is not None else None
            visualize_trajectories(
                trajectories,
                output_path,
                selected_topics=args.topics,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                title=args.title,
                show_legend=not args.no_legend,
            )

            print("\n=== Trajectory Statistics ===")
            for topic, data in trajectories.items():
                positions = data['positions']
                if len(positions) > 0:
                    print(f"\n{topic}:")
                    print(f"  Points: {len(positions)}")
                    print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
                    print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
                    print(f"  Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
                    if len(data['times']) > 0:
                        duration = data['times'][-1] - data['times'][0]
                        print(f"  Duration: {duration:.2f} s")
                        print(f"  Time range: [{data['times'][0]:.2f}, {data['times'][-1]:.2f}] s")
        else:
            print("Failed to extract trajectory data")

if __name__ == '__main__':
    main()

