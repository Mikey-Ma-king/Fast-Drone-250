#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，只保存图片
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
import threading
import time
import os
import math

class PositionVelocityScatter:
    def __init__(self):
        rospy.init_node('position_velocity_scatter', anonymous=True)
        
        # 数据存储
        self.vins_positions = []  # vins fusion位置数据
        self.dog_velocities = []  # dog pos processed速度数据
        self.target_positions = []  # target ekf pose位置数据
        self.data_lock = threading.Lock()  # 线程锁
        
        # 最新数据存储
        self.latest_vins_pos = None
        self.latest_dog_vel = None
        self.latest_target_pos = None
        self.latest_vins_yaw = None  # 飞机yaw角度
        
        # 坐标轴选择配置 (0=x, 1=y, 2=z) - 统一使用一个参数
        self.coordinate_axis = 0  # 默认x轴
        
        # 平均速度参数
        self.velocity_average_points = 1  # 往前a个点的平均速度，默认5个点
        
        # 频率配置
        self.data_collection_freq = 10.0  # 数据采集频率 (Hz) - 默认10Hz (0.1s)
        self.image_save_freq = 1.0  # 图片保存频率 (Hz) - 默认1Hz (1.0s)
        
        # 订阅话题
        self.vins_pose_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_pose_callback)
        self.dog_vel_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self.dog_vel_callback)
        self.target_pose_sub = rospy.Subscriber('/target_ekf_odom', Odometry, self.target_pose_callback)
        
        # 设置matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 动态设置坐标轴标签
        axis_names = ['X', 'Y', 'Z']
        self.ax.set_xlabel(f'Dog Processed Average Velocity {axis_names[self.coordinate_axis]} (m/s)', fontsize=12)
        self.ax.set_ylabel(f'Target EKF Position {axis_names[self.coordinate_axis]} - VINS Fusion Position {axis_names[self.coordinate_axis]} (m)', fontsize=12)
        self.ax.set_title('Position Difference vs Dog Velocity Scatter Plot', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
        # 初始化散点图和拟合曲线
        self.scatter = self.ax.scatter([], [], c='blue', alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        self.fit_line = None  # 拟合曲线对象
        
        # 设置坐标轴范围
        self.ax.set_xlim(-2, 2)  # 速度范围
        self.ax.set_ylim(-2, 2)  # 位置差范围
        
        # 图片保存路径（动态生成）
        self.base_path = '/home/pc/Fast-Drone-250/scatter_plot'
        self.image_path = self.generate_image_path()
        
        # 数据点限制（避免内存过多）
        self.max_points = 1000
        
        # 启动数据采集和图片保存线程
        self.start_data_collection()
        self.start_image_saving()
        
    def generate_image_path(self):
        """根据坐标轴生成图片路径"""
        axis_names = ['X', 'Y', 'Z']
        filename = f"{self.base_path}_Axis{axis_names[self.coordinate_axis]}.png"
        return filename
        
    def set_coordinate_axis(self, axis=0):
        """设置坐标轴选择"""
        self.coordinate_axis = axis  # 0=x, 1=y, 2=z
        
        # 更新坐标轴标签
        axis_names = ['X', 'Y', 'Z']
        self.ax.set_xlabel(f'Dog Processed Average Velocity {axis_names[self.coordinate_axis]} (m/s)', fontsize=12)
        self.ax.set_ylabel(f'Target EKF Position {axis_names[self.coordinate_axis]} - VINS Fusion Position {axis_names[self.coordinate_axis]} (m)', fontsize=12)
        
        # 更新图片路径
        self.image_path = self.generate_image_path()
        
        rospy.loginfo(f"Coordinate axis set: {axis_names[self.coordinate_axis]}")
        rospy.loginfo(f"Image path updated: {self.image_path}")
        
    def set_velocity_average_points(self, points=5):
        """设置平均速度的点数"""
        self.velocity_average_points = max(1, points)  # 至少1个点
        rospy.loginfo(f"Velocity average points set: {self.velocity_average_points}")
        
    def set_frequencies(self, data_collection_freq=10.0, image_save_freq=1.0):
        """设置频率参数"""
        self.data_collection_freq = data_collection_freq  # Hz
        self.image_save_freq = image_save_freq  # Hz
        
        rospy.loginfo(f"Frequencies set: Data collection={self.data_collection_freq}Hz, Image save={self.image_save_freq}Hz")
        
    def rotate_to_aircraft_frame(self, x, y, yaw):
        """将世界坐标系转换为飞机机头坐标系"""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # 旋转矩阵：将世界坐标系转换为飞机机头坐标系
        # 使用与traj_server.cpp相同的转换逻辑
        rotated_x = cos_yaw * x - sin_yaw * y
        rotated_y = sin_yaw * x + cos_yaw * y
        
        return rotated_x, rotated_y
    
    def calculate_average_velocity(self, dog_vel, vins_pos, target_pos):
        """计算从当前点往前a个点的平均速度"""
        n_points = len(dog_vel)
        if n_points < self.velocity_average_points:
            # 如果数据点不够，返回当前速度
            if self.coordinate_axis == 2:  # Z轴不需要旋转
                return dog_vel[:, self.coordinate_axis]
            else:  # X和Y轴需要旋转
                current_yaw = self.latest_vins_yaw
                dog_vel_x = dog_vel[:, 0]
                dog_vel_y = dog_vel[:, 1]
                rotated_vel_x, rotated_vel_y = self.rotate_to_aircraft_frame(dog_vel_x, dog_vel_y, current_yaw)
                
                if self.coordinate_axis == 0:  # X轴
                    return rotated_vel_x
                else:  # Y轴
                    return rotated_vel_y
        
        # 计算平均速度
        avg_velocities = []
        for i in range(n_points):
            # 从当前点往前取a个点
            start_idx = max(0, i - self.velocity_average_points + 1)
            end_idx = i + 1
            
            # 取这a个点的速度数据
            if self.coordinate_axis == 2:  # Z轴不需要旋转
                vel_segment = dog_vel[start_idx:end_idx, self.coordinate_axis]
            else:  # X和Y轴需要旋转
                current_yaw = self.latest_vins_yaw
                vel_x_segment = dog_vel[start_idx:end_idx, 0]
                vel_y_segment = dog_vel[start_idx:end_idx, 1]
                rotated_vel_x, rotated_vel_y = self.rotate_to_aircraft_frame(vel_x_segment, vel_y_segment, current_yaw)
                
                if self.coordinate_axis == 0:  # X轴
                    vel_segment = rotated_vel_x
                else:  # Y轴
                    vel_segment = rotated_vel_y
            
            # 计算平均值
            avg_vel = np.mean(vel_segment)
            avg_velocities.append(avg_vel)
        
        return np.array(avg_velocities)
        
    def vins_pose_callback(self, msg):
        """VINS Fusion位置和yaw回调"""
        with self.data_lock:
            # 存储位置数据
            self.latest_vins_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            
            # 从四元数提取yaw角度
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            
            # 四元数转yaw角
            self.latest_vins_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    
    def dog_vel_callback(self, msg):
        """Dog Pos Processed速度回调"""
        with self.data_lock:
            # 存储速度数据
            self.latest_dog_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
    
    def target_pose_callback(self, msg):
        """Target EKF位置回调"""
        with self.data_lock:
            # 存储位置数据
            self.latest_target_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
    
    def fit_linear(self, x, y):
        """拟合一次函数 y = ax + b (不过原点)"""
        if len(x) < 2:  # 一次函数至少需要2个点
            return None, None, None
        
        # 一次函数拟合：y = ax + b
        # 构建设计矩阵 X = [x, 1]
        X = np.column_stack([x, np.ones(len(x))])
        
        # 最小二乘法求解：X * [a, b] = y
        # 使用正规方程：(X^T * X)^(-1) * X^T * y
        try:
            coeffs = np.linalg.solve(X.T @ X, X.T @ y)
            a, b = coeffs
        except np.linalg.LinAlgError:
            return None, None, None
        
        # 计算R²
        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return a, b, r_squared
    
    def data_collection_loop(self):
        """数据采集循环"""
        while not rospy.is_shutdown():
            with self.data_lock:
                # 检查是否有新数据
                if (self.latest_vins_pos is not None and 
                    self.latest_dog_vel is not None and 
                    self.latest_target_pos is not None and
                    self.latest_vins_yaw is not None):
                    
                    # 添加数据点
                    self.vins_positions.append(self.latest_vins_pos)
                    self.dog_velocities.append(self.latest_dog_vel)
                    self.target_positions.append(self.latest_target_pos)
                    
                    # 限制数据点数量
                    if len(self.vins_positions) > self.max_points:
                        self.vins_positions.pop(0)
                    if len(self.dog_velocities) > self.max_points:
                        self.dog_velocities.pop(0)
                    if len(self.target_positions) > self.max_points:
                        self.target_positions.pop(0)
            
            time.sleep(1.0 / self.data_collection_freq)  # 根据设置的频率采集
    
    def image_saving_loop(self):
        """图片保存循环"""
        while not rospy.is_shutdown():
            with self.data_lock:
                # 检查是否有数据
                min_length = min(len(self.vins_positions), 
                               len(self.dog_velocities), 
                               len(self.target_positions))
                
                if min_length < 1:
                    self.ax.set_title('Position Difference vs Dog Velocity Scatter Plot (Waiting for data...)')
                else:
                    # 取相同长度的数据
                    vins_pos = np.array(self.vins_positions[-min_length:])
                    dog_vel = np.array(self.dog_velocities[-min_length:])
                    target_pos = np.array(self.target_positions[-min_length:])
                    
                    # 计算平均速度
                    dog_data = self.calculate_average_velocity(dog_vel, vins_pos, target_pos)
                    
                    # 计算位置差
                    if self.coordinate_axis == 2:  # Z轴不需要旋转
                        pos_diff = target_pos[:, self.coordinate_axis] - vins_pos[:, self.coordinate_axis]
                    else:  # X和Y轴需要旋转到飞机机头坐标系
                        # 获取对应的yaw角度（使用最新的yaw）
                        current_yaw = self.latest_vins_yaw
                        
                        # 旋转位置差到飞机机头坐标系
                        pos_diff_x = target_pos[:, 0] - vins_pos[:, 0]
                        pos_diff_y = target_pos[:, 1] - vins_pos[:, 1]
                        rotated_pos_x, rotated_pos_y = self.rotate_to_aircraft_frame(pos_diff_x, pos_diff_y, current_yaw)
                        
                        if self.coordinate_axis == 0:  # X轴
                            pos_diff = rotated_pos_x
                        else:  # Y轴
                            pos_diff = rotated_pos_y
                    
                    # 更新散点图数据
                    if len(dog_data) > 0:
                        self.scatter.set_offsets(np.column_stack((dog_data, pos_diff)))
                        
                        # 动态调整坐标轴范围
                        if len(dog_data) > 1:
                            vel_range = max(np.max(np.abs(dog_data)) * 1.2, 0.1)
                            pos_range = max(np.max(np.abs(pos_diff)) * 1.2, 0.1)
                            
                            self.ax.set_xlim(-vel_range, vel_range)
                            self.ax.set_ylim(-pos_range, pos_range)
                        
                        # 拟合一次函数
                        if len(dog_data) > 1:
                            a, b, r_squared = self.fit_linear(dog_data, pos_diff)
                            
                            if a is not None:
                                # 绘制拟合一次函数
                                x_line = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], 100)
                                y_line = a * x_line + b
                                
                                # 删除旧的拟合曲线
                                if self.fit_line is not None:
                                    self.fit_line.remove()
                                
                                # 绘制新的拟合一次函数
                                axis_names = ['X', 'Y', 'Z']
                                self.fit_line, = self.ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, 
                                                           label=f'y = {a:.3f}x + {b:.3f}')
                                
                                # 更新标题和添加图例
                                title = f'Position Difference vs Dog Velocity Scatter Plot\n'
                                title += f'Axis: {axis_names[self.coordinate_axis]} (Aircraft Frame)\n'
                                title += f'Freq: Collect={self.data_collection_freq}Hz, Save={self.image_save_freq}Hz\n'
                                title += f'Fitted Linear: y = {a:.3f}x + {b:.3f}, R² = {r_squared:.3f}, Points: {len(dog_data)}, Avg: {self.velocity_average_points}'
                                self.ax.set_title(title, fontsize=9)
                                
                                # 添加图例
                                self.ax.legend(loc='upper right')
                                
                                # 打印参数到终端
                                rospy.loginfo(f"Axis: {axis_names[self.coordinate_axis]} (Aircraft Frame)")
                                rospy.loginfo(f"Freq: Collect={self.data_collection_freq}Hz, Save={self.image_save_freq}Hz")
                                rospy.loginfo(f"Fitted linear: y = {a:.3f}x + {b:.3f}, R² = {r_squared:.3f}, Avg points: {self.velocity_average_points}")
                            else:
                                axis_names = ['X', 'Y', 'Z']
                                self.ax.set_title(f'Position Difference vs Dog Velocity Scatter Plot\n'
                                                f'Axis: {axis_names[self.coordinate_axis]} (Aircraft Frame)\n'
                                                f'Freq: Collect={self.data_collection_freq}Hz, Save={self.image_save_freq}Hz\n'
                                                f'Points: {len(dog_data)}')
                        else:
                            axis_names = ['X', 'Y', 'Z']
                            self.ax.set_title(f'Position Difference vs Dog Velocity Scatter Plot\n'
                                            f'Axis: {axis_names[self.coordinate_axis]} (Aircraft Frame)\n'
                                            f'Freq: Collect={self.data_collection_freq}Hz, Save={self.image_save_freq}Hz\n'
                                            f'Points: {len(dog_data)}')
                    else:
                        self.ax.set_title('Position Difference vs Dog Velocity Scatter Plot (No valid data)')
                
                # 保存图片
                try:
                    self.fig.savefig(self.image_path, dpi=100, bbox_inches='tight')
                    rospy.loginfo(f"Image saved: {self.image_path}")
                except Exception as e:
                    rospy.logwarn(f"Save image error: {e}")
            
            time.sleep(1.0 / self.image_save_freq)  # 根据设置的频率保存
    
    def start_data_collection(self):
        """启动数据采集线程"""
        collection_thread = threading.Thread(target=self.data_collection_loop)
        collection_thread.daemon = True
        collection_thread.start()
    
    def start_image_saving(self):
        """启动图片保存线程"""
        saving_thread = threading.Thread(target=self.image_saving_loop)
        saving_thread.daemon = True
        saving_thread.start()
    
    def run(self):
        """运行节点"""
        rospy.loginfo("Position Velocity Scatter Plot Node Started")
        rospy.loginfo("Subscribing to topics:")
        rospy.loginfo("  - /vins_fusion/imu_propagate")
        rospy.loginfo("  - /dog_pos_processed")
        rospy.loginfo("  - /target_ekf_odom")
        rospy.loginfo(f"Data collection frequency: {self.data_collection_freq} Hz")
        rospy.loginfo(f"Image saving frequency: {self.image_save_freq} Hz")
        rospy.loginfo("Real-time linear fitting: y = ax + b (not through origin)")
        rospy.loginfo("Aircraft frame coordinate system (rotated by aircraft yaw)")
        rospy.loginfo("Configurable coordinate axis: 0=X, 1=Y, 2=Z")
        axis_names = ['X', 'Y', 'Z']
        rospy.loginfo(f"Current axis: {axis_names[self.coordinate_axis]} (Aircraft Frame)")
        rospy.loginfo(f"Image will be saved to: {self.image_path}")
        rospy.loginfo("To change axis: scatter_plot.set_coordinate_axis(axis)")
        rospy.loginfo("To change frequencies: scatter_plot.set_frequencies(data_collection_freq, image_save_freq)")
        rospy.loginfo("To change average points: scatter_plot.set_velocity_average_points(points)")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Node shutdown")
        finally:
            plt.close('all')

if __name__ == '__main__':
    try:
        scatter_plot = PositionVelocityScatter()
        
        # 示例：设置不同的参数
        # scatter_plot.set_coordinate_axis(0)  # X轴 (飞机机头方向)
        # scatter_plot.set_coordinate_axis(1)  # Y轴 (飞机左侧方向)
        # scatter_plot.set_coordinate_axis(2)  # Z轴 (飞机上方方向)
        
        # scatter_plot.set_frequencies(data_collection_freq=20.0, image_save_freq=2.0)  # 20Hz采集, 2Hz保存
        # scatter_plot.set_frequencies(data_collection_freq=5.0, image_save_freq=0.5)   # 5Hz采集, 0.5Hz保存
        
        # scatter_plot.set_velocity_average_points(3)   # 往前3个点的平均速度
        # scatter_plot.set_velocity_average_points(10)  # 往前10个点的平均速度
        
        scatter_plot.run()
    except rospy.ROSInterruptException:
        pass