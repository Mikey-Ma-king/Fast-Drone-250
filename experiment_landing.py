#!/usr/bin/env python3
"""
实验脚本：自动运行50次降落实验
流程：
1. 运行 fake_target 脚本
2. 启动 rosbag record
3. 等待随机时间（3-15秒）
4. 发布返航触发信号
5. 监控无人机位置，当高度=0.4时记录误差和耗时
6. 停止 rosbag record
7. 停止 fake_target 脚本
8. 发布停止触发信号
9. 发送 position_cmd 让飞机复位到 (0, 0, 0.5)
10. 等待5秒
11. 重复50次
12. 保存结果到文件（每次实验单独文件夹）
"""

import rospy
import subprocess
import time
import random
import math
import numpy as np
import signal
import os
import threading
import sys
import json
from datetime import datetime
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import Point

# 全局变量：存储当前活动的子进程
active_processes = {
    'fake_target': None,
    'rosbag': None
}
active_processes_lock = threading.Lock()
shutdown_flag = threading.Event()

class UAVPositionListener:
    """监听无人机位置"""
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.pose_callback)
    
    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
    
    def get_position(self):
        """获取当前无人机位置"""
        with self.position_lock:
            return self.position.copy() if self.position is not None else None

class TargetPositionListener:
    """监听目标位置"""
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self.pose_callback)
    
    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
    
    def get_position(self):
        """获取当前目标位置"""
        with self.position_lock:
            return self.position.copy() if self.position is not None else None

class ModeManagerListener:
    """监听mode_manager，记录w值为1或2的时间段"""
    def __init__(self):
        self.mode_value = None
        self.is_recording = False  # 是否在记录模式（w值为1或2）
        self.recording_start_time = None
        self.lock = threading.Lock()
        self.mode_sub = rospy.Subscriber('/mode_manager', PoseStamped, self.mode_callback)
    
    def mode_callback(self, msg):
        with self.lock:
            w_value = msg.pose.orientation.w
            self.mode_value = w_value
            # 如果w值为1或2，开始记录
            if w_value == 1.0 or w_value == 2.0:
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_start_time = msg.header.stamp.to_sec()
                    rospy.loginfo("Mode manager w=%.1f detected, starting recording", w_value)
            else:
                if self.is_recording:
                    self.is_recording = False
                    rospy.loginfo("Mode manager w=%.1f detected, stopping recording", w_value)
    
    def is_recording_mode(self):
        """检查是否在记录模式（w值为1或2）"""
        with self.lock:
            return self.is_recording
    
    def get_recording_start_time(self):
        """获取记录开始时间"""
        with self.lock:
            return self.recording_start_time

class GroundTruthAccelerationListener:
    """监听ground_truth_traj，计算加速度"""
    def __init__(self, mode_listener=None):
        self.position = None
        self.velocity = None
        self.last_velocity = None
        self.last_time = None
        self.acceleration = None
        self.acceleration_magnitudes = []  # 存储所有加速度模值，用于计算均值
        self.mode_listener = mode_listener  # 引用mode_listener，用于检查是否在记录模式
        self.lock = threading.Lock()
        self.odom_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self.odom_callback)
    
    def odom_callback(self, msg):
        with self.lock:
            # 只有在mode_manager为1或2时才记录
            if self.mode_listener is not None and not self.mode_listener.is_recording_mode():
                # 更新速度和位置，但不记录加速度
                self.velocity = [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ]
                self.last_velocity = self.velocity.copy()
                self.last_time = msg.header.stamp.to_sec()
                return
            
            current_time = msg.header.stamp.to_sec()
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            self.velocity = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ]
            
            # 计算加速度（通过速度变化）
            if self.last_velocity is not None and self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0.001:  # 至少1ms间隔
                    acc_x = (self.velocity[0] - self.last_velocity[0]) / dt
                    acc_y = (self.velocity[1] - self.last_velocity[1]) / dt
                    acc_z = (self.velocity[2] - self.last_velocity[2]) / dt
                    self.acceleration = [acc_x, acc_y, acc_z]
                    
                    # 计算加速度模并记录
                    acc_magnitude = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    self.acceleration_magnitudes.append(acc_magnitude)
            
            self.last_velocity = self.velocity.copy()
            self.last_time = current_time
    
    def get_mean_acceleration_magnitude(self):
        """获取加速度模的均值"""
        with self.lock:
            if len(self.acceleration_magnitudes) == 0:
                return 0.0
            return np.mean(self.acceleration_magnitudes)
    
    def reset_acceleration(self):
        """重置加速度记录"""
        with self.lock:
            self.acceleration_magnitudes = []

class LocalizationNoiseListener:
    """监听dog_pos_processed和ground_truth_traj，计算定位噪声"""
    def __init__(self, mode_listener=None):
        self.dog_pos = None
        self.gt_pos = None
        self.noise_magnitudes = []  # 存储所有定位噪声值，用于计算标准差
        self.mode_listener = mode_listener  # 引用mode_listener，用于检查是否在记录模式
        self.lock = threading.Lock()
        self.dog_pos_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self.dog_pos_callback)
        self.gt_pos_sub = rospy.Subscriber('/ground_truth_traj', Odometry, self.gt_pos_callback)
    
    def dog_pos_callback(self, msg):
        with self.lock:
            self.dog_pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            self._update_noise()
    
    def gt_pos_callback(self, msg):
        with self.lock:
            self.gt_pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            self._update_noise()
    
    def _update_noise(self):
        """更新定位噪声（dog_pos_processed偏离ground_truth_traj的距离）"""
        # 只有在mode_manager为1或2时才记录
        if self.mode_listener is not None and not self.mode_listener.is_recording_mode():
            return
        
        if self.dog_pos is not None and self.gt_pos is not None:
            # 计算XY平面的距离（定位噪声）
            noise_x = self.dog_pos[0] - self.gt_pos[0]
            noise_y = self.dog_pos[1] - self.gt_pos[1]
            noise_magnitude = math.sqrt(noise_x**2 + noise_y**2)
            
            # 记录所有噪声值
            self.noise_magnitudes.append(noise_magnitude)
    
    def get_noise_std(self):
        """获取定位噪声的标准差"""
        with self.lock:
            if len(self.noise_magnitudes) == 0:
                return 0.0
            return np.std(self.noise_magnitudes)
    
    def reset_noise(self):
        """重置定位噪声记录"""
        with self.lock:
            self.noise_magnitudes = []

def publish_trigger(pub, w_value):
    """发布触发信号到 /mode_manager"""
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = 0.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 0.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = w_value
    pub.publish(msg)
    rospy.loginfo("Published trigger with w=%.1f", w_value)

def publish_position_cmd(pub, x, y, z, traj_id=1):
    """发布位置命令"""
    cmd = PositionCommand()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "world"
    cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
    cmd.trajectory_id = traj_id
    cmd.position.x = x
    cmd.position.y = y
    cmd.position.z = z
    cmd.velocity.x = 0.0
    cmd.velocity.y = 0.0
    cmd.velocity.z = 0.0
    cmd.acceleration.x = 0.0
    cmd.acceleration.y = 0.0
    cmd.acceleration.z = 0.0
    cmd.yaw = 0.0
    cmd.yaw_dot = 0.0
    pub.publish(cmd)
    rospy.loginfo("Published position command: (%.2f, %.2f, %.2f)", x, y, z)

def wait_for_landing(uav_listener, target_listener, timeout=60.0):
    """
    等待无人机降落（高度达到0.4m）
    同时检测无人机是否停止移动超过5秒
    返回：(landing_time, uav_pos, target_pos, error_x, error_y, error_xy, stopped)
    stopped: True表示因停止移动而结束，False表示正常降落或超时
    """
    start_time = time.time()
    rate = rospy.Rate(50)  # 50Hz
    
    # 用于检测停止移动的变量
    last_position = None
    last_position_time = None
    position_threshold = 0.05  # 位置变化阈值（5cm）
    stop_duration = 5.0  # 停止持续时间阈值（5秒）
    
    while time.time() - start_time < timeout:
        # 检查关闭信号
        if shutdown_flag.is_set():
            rospy.logwarn("Shutdown signal received during landing wait")
            uav_pos = uav_listener.get_position()
            target_pos = target_listener.get_position()
            if uav_pos is not None and target_pos is not None:
                error_x = uav_pos[0] - target_pos[0]
                error_y = uav_pos[1] - target_pos[1]
                error_xy = np.sqrt(error_x**2 + error_y**2)
                return time.time() - start_time, uav_pos, target_pos, error_x, error_y, error_xy, False
            return time.time() - start_time, None, None, None, None, None, False
        
        uav_pos = uav_listener.get_position()
        target_pos = target_listener.get_position()
        
        if uav_pos is not None:
            current_time = time.time()
            
            # 检测停止移动
            if last_position is not None:
                # 计算位置变化（仅考虑XY平面）
                position_change = np.sqrt((uav_pos[0] - last_position[0])**2 + 
                                        (uav_pos[1] - last_position[1])**2)
                
                if position_change < position_threshold:
                    # 位置变化小于阈值，可能停止移动
                    if last_position_time is None:
                        last_position_time = current_time
                    else:
                        # 检查停止持续时间
                        stop_time = current_time - last_position_time
                        if stop_time >= stop_duration:
                            # 停止移动超过5秒，结束实验
                            rospy.logwarn("UAV stopped moving for %.2f seconds (>%.2f s), ending experiment", 
                                        stop_time, stop_duration)
                            if target_pos is not None:
                                error_x = uav_pos[0] - target_pos[0]
                                error_y = uav_pos[1] - target_pos[1]
                                error_xy = np.sqrt(error_x**2 + error_y**2)
                                return time.time() - start_time, uav_pos, target_pos, error_x, error_y, error_xy, True
                            else:
                                return time.time() - start_time, uav_pos, None, None, None, None, True
                else:
                    # 位置变化大于阈值，重置停止计时
                    last_position_time = None
            
            # 更新最后位置
            last_position = uav_pos.copy()
            
            # 检查高度是否达到0.4m（允许0.05m的误差）
            if target_pos is not None and abs(uav_pos[2] - 0.4) < 0.05:
                landing_time = time.time() - start_time
                error_x = uav_pos[0] - target_pos[0]
                error_y = uav_pos[1] - target_pos[1]
                error_xy = np.sqrt(error_x**2 + error_y**2)
                rospy.loginfo("Landing detected! Height=%.3f, Error_X=%.3f m, Error_Y=%.3f m, Error=%.3f m, Time=%.2f s", 
                            uav_pos[2], error_x, error_y, error_xy, landing_time)
                return landing_time, uav_pos, target_pos, error_x, error_y, error_xy, False
        
        rate.sleep()
    
    # 超时
    rospy.logwarn("Landing timeout after %.1f seconds", timeout)
    uav_pos = uav_listener.get_position()
    target_pos = target_listener.get_position()
    if uav_pos is not None and target_pos is not None:
        error_x = uav_pos[0] - target_pos[0]
        error_y = uav_pos[1] - target_pos[1]
        error_xy = np.sqrt(error_x**2 + error_y**2)
        return timeout, uav_pos, target_pos, error_x, error_y, error_xy, False
    return timeout, None, None, None, None, None, False

def cleanup_processes():
    """清理所有活动的子进程"""
    global active_processes
    with active_processes_lock:
        rospy.loginfo("Cleaning up subprocesses...")
        
        # 停止 rosbag
        if active_processes['rosbag'] is not None:
            try:
                rospy.loginfo("Stopping rosbag process...")
                active_processes['rosbag'].send_signal(signal.SIGINT)
                try:
                    active_processes['rosbag'].wait(timeout=3)
                except subprocess.TimeoutExpired:
                    active_processes['rosbag'].kill()
                    active_processes['rosbag'].wait()
                rospy.loginfo("Rosbag process stopped")
            except Exception as e:
                rospy.logwarn("Error stopping rosbag: %s", str(e))
            finally:
                active_processes['rosbag'] = None
        
        # 停止 fake_target
        if active_processes['fake_target'] is not None:
            try:
                rospy.loginfo("Stopping fake_target process...")
                active_processes['fake_target'].terminate()
                try:
                    active_processes['fake_target'].wait(timeout=3)
                except subprocess.TimeoutExpired:
                    active_processes['fake_target'].kill()
                    active_processes['fake_target'].wait()
                rospy.loginfo("Fake_target process stopped")
            except Exception as e:
                rospy.logwarn("Error stopping fake_target: %s", str(e))
            finally:
                active_processes['fake_target'] = None

def signal_handler(signum, frame):
    """信号处理函数（Ctrl+C）"""
    rospy.logwarn("\nReceived interrupt signal (Ctrl+C). Cleaning up...")
    shutdown_flag.set()
    cleanup_processes()
    rospy.loginfo("Cleanup completed. Exiting...")
    sys.exit(0)

def clear_fake_target_params():
    """清空fake_target节点下的所有参数"""
    # 可能的命名空间
    possible_namespaces = [
        '/target_triangle_sensor_publisher',
        '/fake_target',
        ''
    ]
    
    # 需要清空的参数列表
    param_names = [
        'bumpy_area_x_min',
        'bumpy_area_x_max',
        'bumpy_area_y_min',
        'bumpy_area_y_max'
    ]
    
    cleared_count = 0
    for ns in possible_namespaces:
        for param_name in param_names:
            try:
                if ns:
                    full_param_name = f'{ns}/{param_name}'
                else:
                    full_param_name = param_name
                
                # 检查参数是否存在
                if rospy.has_param(full_param_name):
                    rospy.delete_param(full_param_name)
                    cleared_count += 1
                    rospy.loginfo("Deleted parameter: %s", full_param_name)
            except Exception as e:
                # 参数不存在或其他错误，忽略
                continue
    
    if cleared_count > 0:
        rospy.loginfo("Cleared %d parameters from fake_target node namespaces", cleared_count)
    else:
        rospy.loginfo("No parameters to clear (they may not exist)")

def get_bumpy_area_from_params():
    """从ROS参数服务器获取颠簸区域信息"""
    # 尝试多个可能的命名空间
    possible_namespaces = [
        '/target_triangle_sensor_publisher',
        '/fake_target',
        ''
    ]
    
    for ns in possible_namespaces:
        try:
            if ns:
                param_prefix = f'{ns}/'
            else:
                param_prefix = ''
            
            # 尝试获取参数（使用 get_param 的默认值机制）
            try:
                bumpy_area_x_min = rospy.get_param(f'{param_prefix}bumpy_area_x_min')
                bumpy_area_x_max = rospy.get_param(f'{param_prefix}bumpy_area_x_max')
                bumpy_area_y_min = rospy.get_param(f'{param_prefix}bumpy_area_y_min')
                bumpy_area_y_max = rospy.get_param(f'{param_prefix}bumpy_area_y_max')
                
                # 如果所有值都不是None，说明找到了
                if all(v is not None for v in [bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max]):
                    return {
                        'x_min': float(bumpy_area_x_min),
                        'x_max': float(bumpy_area_x_max),
                        'y_min': float(bumpy_area_y_min),
                        'y_max': float(bumpy_area_y_max)
                    }
            except KeyError:
                continue
        except Exception as e:
            continue
    
    return None

def run_single_experiment(experiment_num, uav_listener, target_listener, trigger_pub, pos_cmd_pub, base_dir):
    """运行单次实验"""
    # 创建监听器用于记录mode_manager为1或2时的数据
    mode_listener = ModeManagerListener()
    gt_acc_listener = GroundTruthAccelerationListener(mode_listener=mode_listener)
    noise_listener = LocalizationNoiseListener(mode_listener=mode_listener)
    
    # 等待监听器初始化
    time.sleep(1.0)
    
    # 重置记录的加速度和噪声数据
    gt_acc_listener.reset_acceleration()
    noise_listener.reset_noise()
    rospy.loginfo("=" * 60)
    rospy.loginfo("Starting experiment %d/%d", experiment_num, 50)
    rospy.loginfo("=" * 60)
    
    # 创建实验文件夹
    exp_dir = os.path.join(base_dir, f"experiment_{experiment_num:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    rospy.loginfo("Experiment directory: %s", exp_dir)
    
    # 检查是否收到关闭信号
    if shutdown_flag.is_set():
        return None
    
    # 0. 清空fake_target节点下的参数，确保每次实验开始时参数是干净的
    rospy.loginfo("Step 0: Clearing fake_target node parameters...")
    clear_fake_target_params()
    
    # 1. 运行 fake_target 脚本
    rospy.loginfo("Step 1: Starting fake_target script...")
    # 为每次实验设置不同的随机种子，确保随机生成的颠簸区域不同
    # 使用实验编号和时间戳组合生成唯一的种子
    seed_value = (experiment_num * 1000000 + int(time.time() * 1000)) % (2**32)
    env = os.environ.copy()
    env['FAKE_TARGET_RANDOM_SEED'] = str(seed_value)
    rospy.loginfo("Setting random seed for fake_target: %d", seed_value)
    fake_target_process = subprocess.Popen(
        ['python3', '/home/pc/Fast-Drone-250/fake_target.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    with active_processes_lock:
        active_processes['fake_target'] = fake_target_process
    time.sleep(3)  # 等待脚本启动并初始化参数
    
    # 再次检查关闭信号
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    
    # 获取颠簸区域信息（尝试多次，因为可能是随机生成的）
    bumpy_area = None
    for attempt in range(5):
        bumpy_area = get_bumpy_area_from_params()
        if bumpy_area:
            break
        time.sleep(0.5)
    
    if bumpy_area:
        rospy.loginfo("Bumpy area: x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                     bumpy_area['x_min'], bumpy_area['x_max'],
                     bumpy_area['y_min'], bumpy_area['y_max'])
    else:
        rospy.logwarn("Could not retrieve bumpy area information")
    
    # 2. 启动 rosbag record
    rospy.loginfo("Step 2: Starting rosbag record...")
    bag_filename = os.path.join(exp_dir, f"experiment_{experiment_num:03d}.bag")
    topics = [
        '/dog_pos',
        '/mode_manager',
        '/vins_fusion/imu_propagate',
        '/position_cmd',
        '/target_ekf_odom',
        '/dog_pos_processed',
        '/drone2/planning/traj',
        '/traj_v',
        '/ground_truth_traj'
    ]
    
    rosbag_process = subprocess.Popen(
        ['rosbag', 'record', '-O', bag_filename] + topics,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    with active_processes_lock:
        active_processes['rosbag'] = rosbag_process
    time.sleep(1)  # 等待 rosbag 启动
    
    # 检查关闭信号
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    
    # 3. 等待随机时间（5-35秒）
    wait_time = random.uniform(5.0, 35.0)
    rospy.loginfo("Step 3: Waiting %.2f seconds...", wait_time)
    # 分段等待，以便能够响应关闭信号
    elapsed = 0.0
    while elapsed < wait_time and not shutdown_flag.is_set():
        sleep_interval = min(0.5, wait_time - elapsed)
        time.sleep(sleep_interval)
        elapsed += sleep_interval
    
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    
    # 4. 发布返航触发信号
    rospy.loginfo("Step 4: Publishing return trigger...")
    publish_trigger(trigger_pub, 0.0)
    time.sleep(0.5)  # 确保消息发布
    
    # 5. 监控无人机位置，等待降落
    rospy.loginfo("Step 5: Monitoring landing...")
    landing_time, uav_pos, target_pos, error_x, error_y, error_xy, stopped = wait_for_landing(uav_listener, target_listener)
    
    # 判断实验成功或失败
    # 成功条件：误差小于10cm（0.1m）且未因停止移动而结束
    # 如果stopped为True，则success为False
    success = False
    if not stopped and error_xy is not None and error_xy < 0.1:
        success = True
        rospy.loginfo("Experiment SUCCESS: Error=%.3f m < 0.1 m", error_xy)
    else:
        if stopped:
            rospy.logwarn("Experiment FAILED: UAV stopped moving")
        elif error_xy is not None:
            rospy.logwarn("Experiment FAILED: Error=%.3f m >= 0.1 m", error_xy)
        else:
            rospy.logwarn("Experiment FAILED: No valid error data")
    
    if shutdown_flag.is_set():
        cleanup_processes()
        return None
    
    # 6. 停止 rosbag record
    rospy.loginfo("Step 6: Stopping rosbag record...")
    with active_processes_lock:
        if active_processes['rosbag'] is not None:
            active_processes['rosbag'].send_signal(signal.SIGINT)
            try:
                active_processes['rosbag'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['rosbag'].kill()
            active_processes['rosbag'] = None
    time.sleep(1)
    
    # 7. 停止 fake_target 脚本
    rospy.loginfo("Step 7: Stopping fake_target script...")
    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            active_processes['fake_target'].terminate()
            try:
                active_processes['fake_target'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['fake_target'].kill()
            active_processes['fake_target'] = None
    time.sleep(1)
    
    # 8. 发布停止触发信号
    rospy.loginfo("Step 8: Publishing stop trigger...")
    publish_trigger(trigger_pub, -1.0)
    time.sleep(0.5)
    
    # 9. 发送 position_cmd 让飞机复位到 (0, 0, 0.5)，使用控制序列逐步到达
    rospy.loginfo("Step 9: Sending reset position command sequence...")
    
    # 获取当前无人机位置
    current_pos = uav_listener.get_position()
    if current_pos is None:
        rospy.logwarn("Cannot get current UAV position, using direct reset")
        current_pos = [0.0, 0.0, 0.5]
    
    current_x, current_y, current_z = current_pos[0], current_pos[1], current_pos[2]
    target_x, target_y, target_z = 0.0, 0.0, 0.5
    
    # 计算距离
    distance_xy = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
    
    # 如果距离较远，使用中间点序列；如果距离较近，直接到达
    if distance_xy > 2.0:  # 距离超过2米，使用中间点
        # 创建中间点序列：先到中间点，再到目标点
        mid_x = current_x * 0.5  # 中间点x坐标（当前位置的一半）
        mid_y = current_y * 0.5  # 中间点y坐标（当前位置的一半）
        mid_z = max(current_z, 1.0)  # 中间点高度（至少1米，确保安全）
        
        # 第一步：先到中间点（高度先提升到安全高度）
        rospy.loginfo("Reset step 1: Moving to intermediate position (%.2f, %.2f, %.2f)...", mid_x, mid_y, mid_z)
        for _ in range(20):  # 发布多次确保收到
            publish_position_cmd(pos_cmd_pub, mid_x, mid_y, mid_z)
            time.sleep(0.1)
        time.sleep(3.0)  # 等待到达中间点
        
        # 第二步：从中间点到目标点
        rospy.loginfo("Reset step 2: Moving to target position (%.2f, %.2f, %.2f)...", target_x, target_y, target_z)
        for _ in range(20):
            publish_position_cmd(pos_cmd_pub, target_x, target_y, target_z)
            time.sleep(0.1)
        time.sleep(3.0)  # 等待到达目标点
    else:
        # 距离较近，直接到达目标点
        rospy.loginfo("Reset: Moving directly to target position (%.2f, %.2f, %.2f)...", target_x, target_y, target_z)
        for _ in range(20):
            publish_position_cmd(pos_cmd_pub, target_x, target_y, target_z)
        time.sleep(0.1)
        time.sleep(3.0)  # 等待到达目标点
    
    # 10. 等待2秒确保稳定
    rospy.loginfo("Step 10: Waiting 2 seconds for stabilization...")
    time.sleep(2.0)
    
    # 获取mode_manager为1或2时的加速度均值和定位噪声标准差
    mean_acc_magnitude = gt_acc_listener.get_mean_acceleration_magnitude()
    noise_std = noise_listener.get_noise_std()
    
    rospy.loginfo("Mode manager recording data:")
    rospy.loginfo("  Mean ground platform acceleration magnitude: %.4f m/s²", mean_acc_magnitude)
    rospy.loginfo("  Localization noise std: %.4f m", noise_std)
    
    status_str = "SUCCESS" if success else "FAILED"
    rospy.loginfo("Experiment %d completed [%s]. Error_X=%.3f m, Error_Y=%.3f m, Error=%.3f m, Time=%.2f s", 
                 experiment_num, status_str,
                 error_x if error_x is not None else -1.0,
                 error_y if error_y is not None else -1.0,
                 error_xy if error_xy is not None else -1.0, 
                 landing_time)
    
    return {
        'experiment_num': experiment_num,
        'waiting_time': wait_time,  # 等待时间（从启动rosbag到发布返航信号之间的等待时间）
        'landing_time': landing_time,  # 降落时间（从发布返航信号到降落完成的时间）
        'success': success,  # success=False 表示失败（包括stopped的情况）
        'error_x': error_x,
        'error_y': error_y,
        'error_xy': error_xy,
        'uav_pos': uav_pos,
        'target_pos': target_pos,
        'bumpy_area': bumpy_area,
        'exp_dir': exp_dir,
        'mean_ground_acceleration_magnitude': mean_acc_magnitude,  # 地面平台加速度模的均值（mode_manager为1或2时，m/s²）
        'localization_noise_std': noise_std  # 定位噪声的标准差（mode_manager为1或2时，dog_pos_processed偏离ground_truth_traj的距离，m）
    }

def main():
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rospy.init_node('experiment_landing', anonymous=True)
    
    # 创建主实验文件夹（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join('/home/pc/Fast-Drone-250', f'experiment_results_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)
    rospy.loginfo("Experiment base directory: %s", base_dir)
    
    # 创建发布器
    trigger_pub = rospy.Publisher('/mode_manager', PoseStamped, queue_size=10)
    pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=10)
    
    # 创建监听器
    uav_listener = UAVPositionListener()
    target_listener = TargetPositionListener()
    
    # 等待ROS节点初始化
    rospy.loginfo("Waiting for ROS topics...")
    time.sleep(2)
    
    # 存储实验结果
    results = []
    
    # 运行50次实验
    for i in range(1, 51):
        # 检查关闭信号
        if shutdown_flag.is_set():
            rospy.logwarn("Shutdown signal received. Stopping experiments...")
            break
        
        try:
            result = run_single_experiment(i, uav_listener, target_listener, trigger_pub, pos_cmd_pub, base_dir)
            if result is None:  # 如果返回None，说明被中断了
                rospy.logwarn("Experiment %d was interrupted", i)
                break
            results.append(result)
            
            # 保存中间结果（每10次保存一次）
            if i % 10 == 0:
                save_results(results, i, base_dir)
        except KeyboardInterrupt:
            rospy.logwarn("Keyboard interrupt received. Cleaning up...")
            cleanup_processes()
            break
        except Exception as e:
            rospy.logerr("Experiment %d failed: %s", i, str(e))
            results.append({
                'experiment_num': i,
                'waiting_time': -1.0,
                'landing_time': -1.0,
                'success': False,
                'error_x': -1.0,
                'error_y': -1.0,
                'error_xy': -1.0,
                'uav_pos': None,
                'target_pos': None,
                'bumpy_area': None,
                'exp_dir': os.path.join(base_dir, f"experiment_{i:03d}"),
                'mean_ground_acceleration_magnitude': -1.0,
                'localization_noise_std': -1.0
            })
    
    # 确保清理所有进程
    cleanup_processes()
    
    # 保存最终结果
    if results:
        save_results(results, len(results), base_dir)
    rospy.loginfo("Experiments completed! Total: %d", len(results))
    
    # 打印统计信息
    valid_results = [r for r in results if r['error_xy'] is not None and r['error_xy'] >= 0]
    success_count = sum(1 for r in results if r.get('success', False))
    
    if valid_results:
        errors_x = [r['error_x'] for r in valid_results if r.get('error_x') is not None]
        errors_y = [r['error_y'] for r in valid_results if r.get('error_y') is not None]
        errors_xy = [r['error_xy'] for r in valid_results]
        landing_times = [r['landing_time'] for r in valid_results if r['landing_time'] is not None and r['landing_time'] >= 0]
        waiting_times = [r.get('waiting_time', -1.0) for r in valid_results if r.get('waiting_time') is not None and r.get('waiting_time', -1.0) >= 0]
        rospy.loginfo("Statistics:")
        rospy.loginfo("  Total experiments: %d", len(results))
        rospy.loginfo("  Successful experiments: %d (%.1f%%)", success_count, 100.0 * success_count / len(results) if len(results) > 0 else 0.0)
        rospy.loginfo("  Failed experiments: %d (%.1f%%)", len(results) - success_count, 100.0 * (len(results) - success_count) / len(results) if len(results) > 0 else 0.0)
        rospy.loginfo("  Valid experiments: %d/%d", len(valid_results), len(results))
        rospy.loginfo("  Error X - Mean: %.3f m, Std: %.3f m, Min: %.3f m, Max: %.3f m", 
                     np.mean(errors_x), np.std(errors_x), np.min(errors_x), np.max(errors_x))
        rospy.loginfo("  Error Y - Mean: %.3f m, Std: %.3f m, Min: %.3f m, Max: %.3f m", 
                     np.mean(errors_y), np.std(errors_y), np.min(errors_y), np.max(errors_y))
        rospy.loginfo("  Error XY (magnitude) - Mean: %.3f m, Std: %.3f m, Min: %.3f m, Max: %.3f m", 
                     np.mean(errors_xy), np.std(errors_xy), np.min(errors_xy), np.max(errors_xy))
        rospy.loginfo("  Landing Time - Mean: %.2f s, Std: %.2f s, Min: %.2f s, Max: %.2f s", 
                     np.mean(landing_times) if landing_times else 0.0, np.std(landing_times) if landing_times else 0.0, 
                     np.min(landing_times) if landing_times else 0.0, np.max(landing_times) if landing_times else 0.0)
        rospy.loginfo("  Waiting Time - Mean: %.2f s, Std: %.2f s, Min: %.2f s, Max: %.2f s", 
                     np.mean(waiting_times) if waiting_times else 0.0, np.std(waiting_times) if waiting_times else 0.0, 
                     np.min(waiting_times) if waiting_times else 0.0, np.max(waiting_times) if waiting_times else 0.0)

def save_results(results, num_experiments, base_dir):
    """保存实验结果到JSON文件"""
    # 准备JSON数据
    json_data = {
        'total_experiments': num_experiments,
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    # 转换每个实验结果为JSON可序列化的格式
    for r in results:
        exp_data = {
            'experiment_num': r['experiment_num'],
            'waiting_time': float(r.get('waiting_time', -1.0)) if r.get('waiting_time') is not None else -1.0,  # 等待时间（从启动rosbag到发布返航信号之间的等待时间）
            'landing_time': float(r['landing_time']) if r['landing_time'] is not None else -1.0,  # 降落时间（从发布返航信号到降落完成的时间）
            'success': bool(r.get('success', False)),  # success=False 表示失败（包括stopped的情况）
            'error_x': float(r['error_x']) if r.get('error_x') is not None else -1.0,
            'error_y': float(r['error_y']) if r.get('error_y') is not None else -1.0,
            'error_xy': float(r['error_xy']) if r['error_xy'] is not None else -1.0,
            'uav_position': {
                'x': float(r['uav_pos'][0]) if r['uav_pos'] is not None and len(r['uav_pos']) > 0 else None,
                'y': float(r['uav_pos'][1]) if r['uav_pos'] is not None and len(r['uav_pos']) > 1 else None,
                'z': float(r['uav_pos'][2]) if r['uav_pos'] is not None and len(r['uav_pos']) > 2 else None
            } if r['uav_pos'] is not None else None,
            'target_position': {
                'x': float(r['target_pos'][0]) if r['target_pos'] is not None and len(r['target_pos']) > 0 else None,
                'y': float(r['target_pos'][1]) if r['target_pos'] is not None and len(r['target_pos']) > 1 else None,
                'z': float(r['target_pos'][2]) if r['target_pos'] is not None and len(r['target_pos']) > 2 else None
            } if r['target_pos'] is not None else None,
            'bumpy_area': r.get('bumpy_area') if r.get('bumpy_area') is not None else None,
            'experiment_directory': r.get('exp_dir', ''),
            'mean_ground_acceleration_magnitude': float(r.get('mean_ground_acceleration_magnitude', -1.0)) if r.get('mean_ground_acceleration_magnitude') is not None else -1.0,  # 地面平台加速度模的均值（mode_manager为1或2时，m/s²）
            'localization_noise_std': float(r.get('localization_noise_std', -1.0)) if r.get('localization_noise_std') is not None else -1.0  # 定位噪声的标准差（mode_manager为1或2时，dog_pos_processed偏离ground_truth_traj的距离，m）
        }
        json_data['experiments'].append(exp_data)
    
    # 计算统计信息
    valid_results = [r for r in results if r['error_xy'] is not None and r['error_xy'] >= 0]
    if valid_results:
        errors_x = [r['error_x'] for r in valid_results if r.get('error_x') is not None]
        errors_y = [r['error_y'] for r in valid_results if r.get('error_y') is not None]
        errors_xy = [r['error_xy'] for r in valid_results]
        landing_times = [r['landing_time'] for r in valid_results if r['landing_time'] is not None and r['landing_time'] >= 0]
        waiting_times = [r.get('waiting_time', -1.0) for r in valid_results if r.get('waiting_time') is not None and r.get('waiting_time', -1.0) >= 0]
        
        json_data['statistics'] = {
            'valid_experiments': len(valid_results),
            'total_experiments': len(results),
            'error_x': {
                'mean': float(np.mean(errors_x)) if errors_x else None,
                'std': float(np.std(errors_x)) if errors_x else None,
                'min': float(np.min(errors_x)) if errors_x else None,
                'max': float(np.max(errors_x)) if errors_x else None
            },
            'error_y': {
                'mean': float(np.mean(errors_y)) if errors_y else None,
                'std': float(np.std(errors_y)) if errors_y else None,
                'min': float(np.min(errors_y)) if errors_y else None,
                'max': float(np.max(errors_y)) if errors_y else None
            },
            'error_xy': {
                'mean': float(np.mean(errors_xy)) if errors_xy else None,
                'std': float(np.std(errors_xy)) if errors_xy else None,
                'min': float(np.min(errors_xy)) if errors_xy else None,
                'max': float(np.max(errors_xy)) if errors_xy else None
            },
            'waiting_time': {
                'mean': float(np.mean(waiting_times)) if waiting_times else None,
                'std': float(np.std(waiting_times)) if waiting_times else None,
                'min': float(np.min(waiting_times)) if waiting_times else None,
                'max': float(np.max(waiting_times)) if waiting_times else None
            },
            'landing_time': {
                'mean': float(np.mean(landing_times)) if landing_times else None,
                'std': float(np.std(landing_times)) if landing_times else None,
                'min': float(np.min(landing_times)) if landing_times else None,
                'max': float(np.max(landing_times)) if landing_times else None
            }
        }
    else:
        json_data['statistics'] = None
    
    # 保存为JSON文件
    json_filename = os.path.join(base_dir, f'experiment_results_{num_experiments}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    rospy.loginfo("Results saved to %s", json_filename)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

