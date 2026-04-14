#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and visualize 3D trajectories from ROS bag files with confidence-based coloring
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
import argparse
import json
import glob

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
            '/dog_pos_processed',          # Dog position
            '/ground_truth_traj'           # Ground truth trajectory
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
        
        # Extract confidence from dog_pos_processed
        confidence_data = []
        confidence_times = []
        
        # First pass: extract confidence values
        for topic_name, msg, t in bag.read_messages(topics=['/dog_pos_processed']):
            try:
                relative_time = t.to_sec() - bag_start_time
                if start_time is not None and relative_time < start_time:
                    continue
                if end_time is not None and relative_time > end_time:
                    continue
                
                # Confidence is stored in twist.twist.angular.z of dog_pos_processed message
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'twist') and hasattr(msg.twist.twist, 'angular'):
                    confidence = msg.twist.twist.angular.z
                    confidence_data.append(confidence)
                    confidence_times.append(relative_time)
                else:
                    print(f"Warning: Cannot access twist.twist.angular.z in dog_pos_processed message")
            except Exception as e:
                print(f"Warning: Error extracting confidence from dog_pos_processed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store confidence data
        if len(confidence_data) > 0:
            trajectories['/dog_pos_processed_confidence'] = {
                'values': np.array(confidence_data),
                'times': np.array(confidence_times)
            }
            print(f"\nExtracted {len(confidence_data)} confidence values")
        
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
        print(f"\nSuccessfully extracted {len([k for k in trajectories.keys() if k != '/dog_pos_processed_confidence'])} trajectories")
        
    except Exception as e:
        print(f"Error: Failed to read bag file: {e}")
        return None
    
    return trajectories

def interpolate_confidence(confidence_times, confidence_values, target_times):
    """
    Interpolate confidence values to match target times
    
    Args:
        confidence_times: Array of confidence timestamps
        confidence_values: Array of confidence values
        target_times: Array of target timestamps
    
    Returns:
        Array of interpolated confidence values
    """
    if len(confidence_times) == 0 or len(confidence_values) == 0:
        return np.ones(len(target_times)) * 0.5  # Default confidence
    
    # Use nearest neighbor interpolation
    interpolated = np.zeros(len(target_times))
    for i, t in enumerate(target_times):
        # Find nearest confidence value
        idx = np.argmin(np.abs(confidence_times - t))
        if abs(confidence_times[idx] - t) < 0.1:  # Within 0.1s
            interpolated[i] = confidence_values[idx]
        else:
            # Use default if too far
            interpolated[i] = 0.5
    
    return interpolated

def lowpass_filter(positions, times, alpha=0.1):
    """
    Apply simple linear low-pass filter to trajectory positions
    
    Args:
        positions: Array of positions (N x 3)
        times: Array of timestamps (N,)
        alpha: Filter coefficient (0-1), smaller values = more smoothing (default: 0.1)
    
    Returns:
        Filtered positions array (N x 3)
    """
    if len(positions) < 2:
        return positions
    
    filtered = np.zeros_like(positions)
    filtered[0] = positions[0]  # Initialize with first value
    
    # Simple linear filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    for i in range(1, len(positions)):
        filtered[i] = alpha * positions[i] + (1 - alpha) * filtered[i-1]
    
    return filtered

def visualize_trajectories(trajectories, output_path=None, selected_topics=None, z_max=None, z_min=None, show_target_lost=False, show_start_end_points=True, disturbed_area=None, position_cmd_filter_cutoff=None, dog_pos_filter_cutoff=None, waiting_time=None):
    """
    Visualize 3D trajectories with confidence-based coloring
    
    Args:
        trajectories: Dictionary of trajectory data
        output_path: Path to save image, if None display image
        selected_topics: List of topics to visualize, if None visualize all
        z_max: Maximum z-axis value
        z_min: Minimum z-axis value
        show_target_lost: If True, show target lost (X marker) instead of landing point (star marker)
        show_start_end_points: If True, show starting point and landing point markers
        disturbed_area: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max' defining the disturbed area, or None to skip
        position_cmd_filter_cutoff: Filter coefficient (0-1) for linear low-pass filter on position_cmd. Smaller values = more smoothing. If None, no filtering is applied.
        dog_pos_filter_cutoff: Filter coefficient (0-1) for linear low-pass filter on dog_pos_processed. Smaller values = more smoothing. If None, no filtering is applied.
        waiting_time: Waiting time (landing_time) in seconds to display in the title, or None to skip
    """
    if not trajectories:
        print("Error: No trajectory data to visualize")
        return
    
    # Filter trajectories based on selection
    if selected_topics is not None:
        filtered_trajectories = {}
        for topic in selected_topics:
            if topic in trajectories:
                filtered_trajectories[topic] = trajectories[topic]
        trajectories = filtered_trajectories
        if not trajectories:
            print("Error: No valid trajectories selected")
            return
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors and labels
    colors = {
        '/vins_fusion/imu_propagate': 'blue',
        '/position_cmd': 'red',  # Will be overridden by confidence coloring
        '/target_ekf_odom': 'green',
        '/dog_pos_processed': 'orange',
        '/ground_truth_traj': 'purple'
    }
    
    labels = {
        '/vins_fusion/imu_propagate': 'Drone Actual Trajectory',
        '/position_cmd': 'Position Command Trajectory',
        '/target_ekf_odom': 'Target Trajectory',
        '/dog_pos_processed': 'Dog Position Trajectory',
        '/ground_truth_traj': 'Ground Truth Trajectory'
    }
    
    # Get confidence data if available
    confidence_times = None
    confidence_values = None
    if '/dog_pos_processed_confidence' in trajectories:
        conf_data = trajectories['/dog_pos_processed_confidence']
        confidence_times = conf_data['times']
        confidence_values = conf_data['values']
        print(f"\nConfidence range: [{confidence_values.min():.3f}, {confidence_values.max():.3f}]")
    
    # Create colormap for confidence (green = high confidence, red = low confidence)
    cmap = LinearSegmentedColormap.from_list('confidence', ['red', 'yellow', 'green'], N=256)
    
    # Flag to track if colorbar has been added
    colorbar_added = False
    
    # Store position_cmd end point for target lost marker
    position_cmd_end_point = None
    
    # Collect all end points for calculating average landing point
    end_points = []
    
    # Plot each trajectory
    for topic, data in trajectories.items():
        if topic == '/dog_pos_processed_confidence':
            continue  # Skip confidence data itself
        
        positions = data['positions']
        times = data['times']
        if len(positions) == 0:
            continue
        
        # Apply low-pass filter to position_cmd if specified
        if topic == '/position_cmd' and position_cmd_filter_cutoff is not None:
            positions = lowpass_filter(positions, times, alpha=position_cmd_filter_cutoff)
            print(f"Applied low-pass filter to /position_cmd with alpha: {position_cmd_filter_cutoff}")
        
        # Apply low-pass filter to dog_pos_processed if specified
        if topic == '/dog_pos_processed' and dog_pos_filter_cutoff is not None:
            positions = lowpass_filter(positions, times, alpha=dog_pos_filter_cutoff)
            print(f"Applied low-pass filter to /dog_pos_processed with alpha: {dog_pos_filter_cutoff}")
        
        color = colors.get(topic, 'gray')
        label = labels.get(topic, topic)
        
        # Special handling for position_cmd with confidence coloring
        if topic == '/position_cmd' and confidence_times is not None and confidence_values is not None:
            # Interpolate confidence to match position_cmd times
            conf_interp = interpolate_confidence(confidence_times, confidence_values, times)
            
            # Normalize confidence to [0, 1] for colormap
            conf_normalized = np.clip(conf_interp, 0.0, 1.0)
            
            # Plot trajectory with color based on confidence
            for i in range(len(positions) - 1):
                color_val = conf_normalized[i]
                rgba = cmap(color_val)
                ax.plot([positions[i, 0], positions[i+1, 0]], 
                       [positions[i, 1], positions[i+1, 1]], 
                       [positions[i, 2], positions[i+1, 2]], 
                       color=rgba, linewidth=2, alpha=0.7)
            
            # Store position_cmd end point for target lost marker
            if len(positions) > 0:
                position_cmd_end_point = positions[-1]
            
            # Add colorbar for confidence (only once)
            if not colorbar_added:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
                cbar.set_label('Confidence (Green=High, Red=Low)', rotation=270, labelpad=20)
                colorbar_added = True
            
            # Add label for position_cmd
            ax.plot([], [], [], color='gray', label=label, linewidth=2)
        else:
            # Plot trajectory line with fixed color
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    color=color, label=label, linewidth=2, alpha=0.7)
        
        # Mark starting point with red circle (if enabled)
        if show_start_end_points and len(positions) > 0:
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      color='red', marker='o', s=150, zorder=10)
            # Collect end point for average landing point
            end_points.append(positions[-1])
        elif len(positions) > 0:
            # Still collect end points even if not showing markers
            end_points.append(positions[-1])
    
    # Plot landing point or target lost (if enabled)
    if show_start_end_points:
        if show_target_lost:
            # Show target lost at position_cmd end point
            if position_cmd_end_point is not None:
                ax.scatter(position_cmd_end_point[0], position_cmd_end_point[1], position_cmd_end_point[2],
                          color='red', marker='x', s=500, linewidths=5, zorder=10, label='Target Lost')
        else:
            # Show landing point at average of all end points
            if len(end_points) > 0:
                end_points_array = np.array(end_points)
                avg_landing_point = np.mean(end_points_array, axis=0)
                ax.scatter(avg_landing_point[0], avg_landing_point[1], avg_landing_point[2],
                          color='red', marker='*', s=500, zorder=10, label='Landing Point')
        
        # Add starting point label to legend (only once)
        ax.scatter([], [], [], color='red', marker='o', s=150, label='Starting Point')
    
    # Draw disturbed area if specified (2D rectangle at height 0.4m)
    if disturbed_area is not None:
        x_min = disturbed_area.get('x_min')
        x_max = disturbed_area.get('x_max')
        y_min = disturbed_area.get('y_min')
        y_max = disturbed_area.get('y_max')
        
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            # Draw 2D rectangle at height 0.4m
            z_height = 0.4
            ax.plot([x_min, x_max, x_max, x_min, x_min], 
                   [y_min, y_min, y_max, y_max, y_min],
                   [z_height, z_height, z_height, z_height, z_height],
                   color='red', linewidth=2, linestyle='--', label='Disturbed Area')
    
    # Set axis labels
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Set title with waiting time if provided
    title = '3D Trajectory Visualization with Confidence'
    if waiting_time is not None:
        title += f' (Waiting Time: {waiting_time:.2f}s)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Set equal aspect ratio for X and Y axes
    all_positions = np.vstack([data['positions'] for topic, data in trajectories.items() 
                              if topic != '/dog_pos_processed_confidence' and len(data['positions']) > 0])
    if len(all_positions) > 0:
        max_range_xy = max(all_positions[:, 0].max() - all_positions[:, 0].min(),
                           all_positions[:, 1].max() - all_positions[:, 1].min()) / 2.0
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        ax.set_xlim(mid_x - max_range_xy, mid_x + max_range_xy)
        ax.set_ylim(mid_y - max_range_xy, mid_y + max_range_xy)
        
        # Set Z-axis range
        if z_max is not None:
            z_axis_max = z_max
        else:
            z_axis_max = all_positions[:, 2].max()
        
        if z_min is not None:
            z_axis_min = z_min
        else:
            z_axis_min = all_positions[:, 2].min()
        
        ax.set_zlim(z_axis_min, z_axis_max)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nImage saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def load_experiment_info_from_json(experiment_dir):
    """
    从实验目录的JSON文件中加载实验信息（颠簸区域、成功/失败状态和等待时间）
    
    Args:
        experiment_dir: 实验目录路径（例如：experiment_results_xxx/experiment_001）
    
    Returns:
        dict: 包含 'bumpy_area'、'success' 和 'landing_time' 的字典，如果找不到则返回None
    """
    # 查找JSON文件（可能在父目录中）
    json_files = []
    
    # 首先在实验目录中查找
    exp_json_files = glob.glob(os.path.join(experiment_dir, '*.json'))
    json_files.extend(exp_json_files)
    
    # 在父目录中查找
    parent_dir = os.path.dirname(experiment_dir)
    parent_json_files = glob.glob(os.path.join(parent_dir, 'experiment_results_*.json'))
    json_files.extend(parent_json_files)
    
    if not json_files:
        print(f"Warning: No JSON files found in {experiment_dir} or parent directory")
        return None
    
    # 读取最新的JSON文件
    json_file = max(json_files, key=os.path.getmtime)
    print(f"Loading experiment info from JSON file: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON中提取当前实验的信息
        if 'experiments' in data:
            # 尝试从实验目录名中提取实验编号
            exp_dir_name = os.path.basename(experiment_dir)
            if exp_dir_name.startswith('experiment_'):
                try:
                    exp_num = int(exp_dir_name.split('_')[1])
                    # 查找对应的实验
                    for exp in data['experiments']:
                        if exp.get('experiment_num') == exp_num:
                            result = {
                                'bumpy_area': exp.get('bumpy_area'),
                                'success': exp.get('success', False),
                                'landing_time': exp.get('landing_time', None)
                            }
                            print(f"Found experiment {exp_num} info: success={result['success']}, landing_time={result['landing_time']:.2f}s, bumpy_area={result['bumpy_area']}")
                            return result
                except (ValueError, IndexError):
                    pass
            
            # 如果找不到，尝试使用第一个实验的信息
            if len(data['experiments']) > 0:
                exp = data['experiments'][0]
                result = {
                    'bumpy_area': exp.get('bumpy_area'),
                    'success': exp.get('success', False),
                    'landing_time': exp.get('landing_time', None)
                }
                print(f"Using first experiment info: success={result['success']}, landing_time={result['landing_time']:.2f}s, bumpy_area={result['bumpy_area']}")
                return result
        
        print("Warning: No experiment info found in JSON file")
        return None
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def load_bumpy_area_from_json(experiment_dir):
    """
    从实验目录的JSON文件中加载颠簸区域信息（保持向后兼容）
    
    Args:
        experiment_dir: 实验目录路径（例如：experiment_results_xxx/experiment_001）
    
    Returns:
        dict: 包含颠簸区域信息的字典，如果找不到则返回None
    """
    info = load_experiment_info_from_json(experiment_dir)
    if info is not None:
        return info.get('bumpy_area')
    return None

def find_bag_file(experiment_dir):
    """
    在实验目录中查找bag文件
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        str: bag文件路径，如果找不到则返回None
    """
    bag_files = glob.glob(os.path.join(experiment_dir, '*.bag'))
    if not bag_files:
        print(f"Error: No bag files found in {experiment_dir}")
        return None
    
    if len(bag_files) > 1:
        print(f"Warning: Multiple bag files found, using: {bag_files[0]}")
    
    return bag_files[0]

def find_all_experiment_dirs(base_dir):
    """
    在实验主目录中查找所有实验子目录
    
    Args:
        base_dir: 实验主目录路径（例如：experiment_results_xxx）
    
    Returns:
        list: 所有实验目录路径的列表
    """
    experiment_dirs = []
    # 查找所有 experiment_XXX 格式的目录
    pattern = os.path.join(base_dir, 'experiment_*')
    dirs = glob.glob(pattern)
    
    # 过滤出目录（排除文件）
    for d in dirs:
        if os.path.isdir(d):
            experiment_dirs.append(d)
    
    # 按实验编号排序
    experiment_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]) if os.path.basename(x).startswith('experiment_') else 0)
    
    return experiment_dirs

def main():
    parser = argparse.ArgumentParser(description='Extract and visualize 3D trajectories from ROS bag files with confidence-based coloring')
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='Path to experiment directory (e.g., experiment_results_xxx/experiment_001). If specified, bag file and bumpy area will be auto-detected.')
    parser.add_argument('--bag', type=str, default=None,
                       help='Path to bag file (ignored if --experiment_dir is specified)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (if not specified, auto-generate from bag filename)')
    parser.add_argument('--start_time', type=float, default=None,
                       help='Start time in seconds (relative to bag start)')
    parser.add_argument('--end_time', type=float, default=None,
                       help='End time in seconds (relative to bag start)')
    parser.add_argument('--topics', type=str, nargs='+', default=None,
                       help='Topics to visualize (e.g., /vins_fusion/imu_propagate /position_cmd). If not specified, visualize all available topics')
    parser.add_argument('--z_max', type=float, default=4.0,
                       help='Maximum z-axis value (default: 4.0 m)')
    parser.add_argument('--z_min', type=float, default=None,
                       help='Minimum z-axis value (default: auto)')
    parser.add_argument('--show_target_lost', action='store_true', default=False,
                       help='Show target lost (X marker) instead of landing point (star marker)')
    parser.add_argument('--show_start_end_points', action='store_true', default=True,
                       help='Show starting point and landing point markers (default: True)')
    parser.add_argument('--hide_start_end_points', action='store_false', dest='show_start_end_points',
                       help='Hide starting point and landing point markers')
    parser.add_argument('--disturbed_area', type=float, nargs=4, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                       default=None,
                       help='Disturbed area boundaries: x_min x_max y_min y_max (in meters). If not specified and --experiment_dir is used, will be auto-loaded from JSON.')
    parser.add_argument('--position_cmd_filter_cutoff', type=float, default=0.1,
                       help='Filter coefficient (0-1) for linear low-pass filter on position_cmd trajectory. Smaller values = more smoothing. If not specified, no filtering is applied.')
    parser.add_argument('--dog_pos_filter_cutoff', type=float, default=None,
                       help='Filter coefficient (0-1) for linear low-pass filter on dog_pos_processed trajectory. Smaller values = more smoothing. If not specified, no filtering is applied.')
    
    args = parser.parse_args()
    
    # 初始化waiting_time变量，用于在单个实验目录模式下传递等待时间
    waiting_time_global = None
    
    args.experiment_dir = '/home/pc/Fast-Drone-250/experiment_results_20260214_171504'
    args.start_time = 1.0
    args.end_time = 100

    # args.topics = ['/vins_fusion/imu_propagate', '/position_cmd', '/target_ekf_odom', '/dog_pos_processed', '/ground_truth_traj','/dog_pos_processed_confidence']
    args.topics = ['/position_cmd', '/ground_truth_traj','/dog_pos_processed_confidence']
    # args.topics = ['/dog_pos_processed','/ground_truth_traj']
    args.show_start_end_points = True
    # args.show_target_lost = False


    # 如果指定了experiment_dir，自动查找bag文件和颠簸区域
    if args.experiment_dir:
        if not os.path.isdir(args.experiment_dir):
            print(f"Error: Experiment directory does not exist: {args.experiment_dir}")
            return
        
        # 检查是主目录还是单个实验目录
        exp_dir_name = os.path.basename(args.experiment_dir)
        is_main_dir = exp_dir_name.startswith('experiment_results')
        
        if is_main_dir:
            # 主目录：查找所有实验子目录
            print(f"Detected main experiment directory: {args.experiment_dir}")
            experiment_dirs = find_all_experiment_dirs(args.experiment_dir)
            print(f"Found {len(experiment_dirs)} experiment directories")
            
            if not experiment_dirs:
                print("Error: No experiment directories found")
                return
            
            # 对每个实验目录进行可视化
            for exp_dir in experiment_dirs:
                print(f"\n{'='*60}")
                print(f"Processing: {os.path.basename(exp_dir)}")
                print(f"{'='*60}")
                
                # 查找bag文件
                bag_file = find_bag_file(exp_dir)
                if bag_file is None:
                    print(f"Skipping {exp_dir}: No bag file found")
                    continue
                
                # 自动加载实验信息（颠簸区域、成功/失败状态和等待时间）
                disturbed_area_list = None
                experiment_success = None
                waiting_time = None
                show_target_lost_for_this_exp = args.show_target_lost  # 默认使用命令行参数
                
                # 尝试从JSON加载实验信息
                exp_info = load_experiment_info_from_json(exp_dir)
                if exp_info is not None:
                    # 加载颠簸区域
                    if args.disturbed_area is None:
                        bumpy_area = exp_info.get('bumpy_area')
                        if bumpy_area is not None:
                            disturbed_area_list = [
                                bumpy_area.get('x_min'),
                                bumpy_area.get('x_max'),
                                bumpy_area.get('y_min'),
                                bumpy_area.get('y_max')
                            ]
                            print(f"Auto-loaded disturbed area: {disturbed_area_list}")
                    
                    # 读取success状态，为每个实验单独设置show_target_lost
                    experiment_success = exp_info.get('success', False)
                    # 如果实验失败，显示target lost（大叉）；如果成功，显示landing point（星）
                    show_target_lost_for_this_exp = not experiment_success
                    print(f"Experiment success={experiment_success}, show_target_lost={show_target_lost_for_this_exp}")
                    
                    # 读取等待时间
                    waiting_time = exp_info.get('landing_time', None)
                    if waiting_time is not None:
                        print(f"Waiting time (landing_time): {waiting_time:.2f}s")
                
                # 设置输出路径（保存到实验目录）
                bag_basename = os.path.splitext(os.path.basename(bag_file))[0]
                output_path = os.path.join(exp_dir, 
                                          f'trajectory_visualization_{bag_basename}.png')
                
                # 提取轨迹数据
                trajectories = extract_trajectory_from_bag(bag_file, 
                                                          start_time=args.start_time,
                                                          end_time=args.end_time)
                
                if trajectories:
                    # 解析颠簸区域
                    disturbed_area = None
                    if disturbed_area_list is not None and len(disturbed_area_list) == 4:
                        if all(v is not None for v in disturbed_area_list):
                            disturbed_area = {
                                'x_min': disturbed_area_list[0],
                                'x_max': disturbed_area_list[1],
                                'y_min': disturbed_area_list[2],
                                'y_max': disturbed_area_list[3]
                            }
                    elif args.disturbed_area is not None and len(args.disturbed_area) == 4:
                        if all(v is not None for v in args.disturbed_area):
                            disturbed_area = {
                                'x_min': args.disturbed_area[0],
                                'x_max': args.disturbed_area[1],
                                'y_min': args.disturbed_area[2],
                                'y_max': args.disturbed_area[3]
                            }
                    
                    # 可视化轨迹
                    visualize_trajectories(trajectories, output_path, selected_topics=args.topics,
                                         z_max=args.z_max, z_min=args.z_min, show_target_lost=show_target_lost_for_this_exp,
                                         show_start_end_points=args.show_start_end_points, disturbed_area=disturbed_area,
                                         position_cmd_filter_cutoff=args.position_cmd_filter_cutoff,
                                         dog_pos_filter_cutoff=args.dog_pos_filter_cutoff,
                                         waiting_time=waiting_time)
                    print(f"Visualization saved to: {output_path}")
                else:
                    print(f"Failed to extract trajectory data for {exp_dir}")
            
            print(f"\n{'='*60}")
            print(f"Completed processing {len(experiment_dirs)} experiments")
            print(f"{'='*60}")
            return
        else:
            # 单个实验目录
            # 查找bag文件
            bag_file = find_bag_file(args.experiment_dir)
            if bag_file is None:
                return
            args.bag = bag_file
            print(f"Using bag file: {args.bag}")
            
            # 自动加载实验信息（颠簸区域、成功/失败状态和等待时间）
            exp_info = load_experiment_info_from_json(args.experiment_dir)
            if exp_info is not None:
                # 加载颠簸区域
                if args.disturbed_area is None:
                    bumpy_area = exp_info.get('bumpy_area')
                    if bumpy_area is not None:
                        args.disturbed_area = [
                            bumpy_area.get('x_min'),
                            bumpy_area.get('x_max'),
                            bumpy_area.get('y_min'),
                            bumpy_area.get('y_max')
                        ]
                        print(f"Auto-loaded disturbed area: {args.disturbed_area}")
                
                # 自动设置show_target_lost（如果未通过命令行指定）
                if not args.show_target_lost:
                    experiment_success = exp_info.get('success', False)
                    # 如果实验失败，显示target lost（大叉）；如果成功，显示landing point（星）
                    args.show_target_lost = not experiment_success
                    print(f"Auto-set show_target_lost={args.show_target_lost} based on experiment success={experiment_success}")
                
                # 读取等待时间并保存到全局变量
                waiting_time_global = exp_info.get('landing_time', None)
                if waiting_time_global is not None:
                    print(f"Waiting time (landing_time): {waiting_time_global:.2f}s")
            
            # 自动设置输出路径（保存到实验目录）
            if args.output is None:
                bag_basename = os.path.splitext(os.path.basename(args.bag))[0]
                args.output = os.path.join(args.experiment_dir, 
                                           f'trajectory_visualization_{bag_basename}.png')
                print(f"Output will be saved to: {args.output}")
    
    # 如果没有指定bag文件，报错
    if args.bag is None:
        print("Error: Either --experiment_dir or --bag must be specified")
        return
    
    if not os.path.exists(args.bag):
        print(f"Error: Bag file does not exist: {args.bag}")
        return
    
    # Extract bag filename and add it to output filename (if not already set)
    if args.output is None:
        bag_basename = os.path.splitext(os.path.basename(args.bag))[0]
        default_output_dir = '/home/pc/Fast-Drone-250'
        args.output = os.path.join(default_output_dir, 
                                   f'trajectory_visualization_confidence_{bag_basename}.png')
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Extract trajectory data
    trajectories = extract_trajectory_from_bag(args.bag, 
                                               start_time=args.start_time,
                                               end_time=args.end_time)
    
    if trajectories:
        # Parse disturbed area if specified
        disturbed_area = None
        if args.disturbed_area is not None and len(args.disturbed_area) == 4:
            # Check if all values are valid
            if all(v is not None for v in args.disturbed_area):
                disturbed_area = {
                    'x_min': args.disturbed_area[0],
                    'x_max': args.disturbed_area[1],
                    'y_min': args.disturbed_area[2],
                    'y_max': args.disturbed_area[3]
                }
            else:
                print("Warning: Invalid disturbed area values, skipping visualization")
        
        # Visualize trajectories
        # 使用全局的waiting_time_global变量（在单个实验目录模式下已设置）
        visualize_trajectories(trajectories, args.output, selected_topics=args.topics,
                             z_max=args.z_max, z_min=args.z_min, show_target_lost=args.show_target_lost,
                             show_start_end_points=args.show_start_end_points, disturbed_area=disturbed_area,
                             position_cmd_filter_cutoff=args.position_cmd_filter_cutoff,
                             dog_pos_filter_cutoff=args.dog_pos_filter_cutoff,
                             waiting_time=waiting_time_global)
        
        # Print statistics
        print("\n=== Trajectory Statistics ===")
        for topic, data in trajectories.items():
            if topic == '/dog_pos_processed_confidence':
                print(f"\n{topic}:")
                print(f"  Confidence values: {len(data['values'])}")
                if len(data['values']) > 0:
                    print(f"  Confidence range: [{data['values'].min():.3f}, {data['values'].max():.3f}]")
                    print(f"  Confidence mean: {data['values'].mean():.3f}")
                continue
            
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

