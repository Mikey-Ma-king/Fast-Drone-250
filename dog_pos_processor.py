#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dog Position Processor Module
处理raw_dog_pos数据，维护yaw offset，发布处理后的dog_pos
"""

import rospy
import numpy as np
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import TakeoffLand
from scipy.spatial.transform import Rotation

class KalmanFilter:
    """简单的卡尔曼滤波器，用于位置和速度滤波"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        初始化卡尔曼滤波器
        
        参数:
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
        """
        # 状态向量: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 100.0  # 初始协方差矩阵
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(6) * process_noise
        
        # 测量噪声协方差矩阵 R
        self.R = np.eye(6) * measurement_noise
        
        # 状态转移矩阵 F (恒定速度模型，将在predict中根据dt更新)
        self.F_base = np.eye(6)  # 基础矩阵
        
        # 观测矩阵 H (直接观测位置和速度)
        self.H = np.eye(6)
        
        self.initialized = False
        self.last_time = None
    
    def predict(self, dt):
        """预测步骤"""
        if not self.initialized:
            return
        
        # 构建状态转移矩阵（恒定速度模型）
        F = self.F_base.copy()
        F[0, 3] = dt  # x = x + vx * dt
        F[1, 4] = dt  # y = y + vy * dt
        F[2, 5] = dt  # z = z + vz * dt
        
        # 预测状态
        self.state = F @ self.state
        
        # 预测协方差
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def update(self, measurement):
        """
        更新步骤
        
        参数:
            measurement: 测量值 [x, y, z, vx, vy, vz]
        """
        if not self.initialized:
            # 首次初始化
            self.state = measurement.copy()
            self.initialized = True
            return
        
        # 计算残差
        residual = measurement - self.H @ self.state
        
        # 计算残差协方差
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # 计算卡尔曼增益
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.state = self.state + K @ residual
        
        # 更新协方差
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance
    
    def filter(self, position, velocity, dt):
        """
        滤波主函数
        
        参数:
            position: 位置 [x, y, z]
            velocity: 速度 [vx, vy, vz]
            dt: 时间间隔
        
        返回:
            滤波后的位置和速度
        """
        # 构建测量向量
        measurement = np.concatenate([position, velocity])
        
        # 预测
        self.predict(dt)
        
        # 更新
        self.update(measurement)
        
        # 返回滤波后的位置和速度
        return self.state[:3].copy(), self.state[3:].copy()
    
    def reset(self):
        """重置滤波器"""
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 100.0
        self.initialized = False
        self.last_time = None

class DogPosProcessor:
    def __init__(self):
        rospy.init_node('dog_pos_processor', anonymous=True)
        
        # Trigger状态
        self.trigger_received = False

        # 参数（完全按照traj_server.cpp的设置）
        self.yaw_filter_gain = 0.1  # yaw offset滤波增益
        self.yaw_stable_threshold = math.radians(5.0)  # 5度，yaw稳定阈值
        self.yaw_exceed_threshold = math.radians(45.0)  # 45度，yaw超限阈值
        self.yaw_exceed_max_count = 5  # yaw超限最大计数
        self.pos_stable_threshold = 0.05  # 5cm，位置稳定阈值
        self.pos_exceed_threshold = 0.3  # 30cm，位置超限阈值
        self.pos_exceed_max_count = 5  # 位置超限最大计数
        self.pos_filter_gain = 0.05  # 位置滤波增益
        self.aoa_pos_filter_gain = 0.05  # AOA位置滤波增益（比pos_filter_gain小）
        self.aoa_pos_step_limit = 0.02  # AOA单次迭代上限（米）
        
        # 前馈系数（参考traj_server.cpp）
        self.camera_offset = 0.37  # 关键参数，与traj_server一致

        # AOA相关参数
        self.aoa_min_distance = 3.0  # m
        
        # 仿真模式：直接输出target_ekf为dog_pos_processed
        self.simulate_mode = False

        # 最终输出的dog相关量存储
        self.final_dog_pos = np.array([0.0, 0.0, 0.0])  # 最终输出的dog位置
        self.final_dog_vel = np.array([0.0, 0.0, 0.0])  # 最终输出的dog速度
        self.final_dog_yaw = 0.0  # 最终输出的dog yaw
        
        # 状态变量
        self.yaw_offset = 0  # yaw offset
        self.pos_offset = np.array([0.0, 0.0, 0.0])  # 位置偏移
        self.precise_yaw_offset_ready = False  # yaw offset是否精确ready
        self.precise_pos_offset_ready = False  # pos offset是否精确ready
        self.yaw_exceed_timer = 0  # yaw超限计时器
        self.pos_exceed_timer = 0  # pos超限计时器
        self.saved_yaw_diff = None  # 保存yaw差值
        self.initialized = False  # 是否已初始化
        
        # AOA增量与状态
        self.aoa_received = False
        self.aoa_count = 0
        self.last_aoa_count = 0
        self.last_aoa_timer = 0
        self.aoa_distance = 0.0  # AOA距离
        self.aoa_angle = 0.0  # AOA角度（机体系下的yaw角）

        # 光流高度（用于修正AOA距离的垂直分量）
        self.flow_z = 0.0
        
        
        # 原始数据
        self.raw_dog_pos = None
        self.raw_dog_vel = np.array([0.0, 0.0, 0.0])
        self.raw_dog_yaw = 0.0
        self.raw_dog_pos_received = False
        self.raw_dog_pos_count = 0
        self.last_raw_dog_pos_count = 0
        self.last_dog_pos_timer = 0
        
        # VINS数据
        self.vins_yaw = 0.0
        self.R_wb = np.eye(3)  # 世界系到机体系的旋转矩阵
        self.vins_pos = np.array([0.0, 0.0, 0.0])  # VINS位置
        self.vins_received = False
        self.vins_count = 0
        self.last_vins_count = 0
        self.last_vins_timer = 0
        
        # 目标数据（来自read模块）
        self.target_dog_yaw = 0.0
        self.target_dog_pos = np.array([0.0, 0.0, 0.0])  # 目标狗位置
        self.target_receive = False
        self.target_count = 0
        self.last_target_count = 0
        self.last_target_timer = 0
        self.last_target_loss_timer = 0
        self.last_target_loss_count = 0
        
        # 初始化标志
        self.dog_vel_initialized = False
        
        # 狗通信角速度相关变量
        self.dog_yaw_rate = 0.0  # 狗通信的角速度
        self.last_dog_yaw_time = 0.0  # 上一帧的时间
        self.yaw_rate_filter_gain = 0.3  # 角速度滤波增益
        self.dog_yaw_rate_initialized = False
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        self.kf_enabled = True  # 是否启用卡尔曼滤波
        self.last_kf_time = None  # 上次滤波时间
        self.yaw_filter_gain_kf = 0.3  # yaw 滤波增益（简单一阶滤波）
        self.filtered_yaw = 0.0  # 滤波后的yaw
        self.yaw_filter_initialized = False
        self.kf_timeout = 1.0  # 卡尔曼滤波超时时间（秒），超过此时间间隔则重置
        
        # 发布者 - 使用Odometry格式
        self.dog_pos_pub = rospy.Publisher(
            '/dog_pos_processed',
            Odometry,
            queue_size=10
        )
        
        # Debug发布者 - 发布AOA计算得到的dog位置
        self.aoa_dog_pos_debug_pub = rospy.Publisher(
            '/dog_pos_aoa_debug',
            Odometry,
            queue_size=10
        )

        # 订阅者
        self.raw_dog_pos_sub = rospy.Subscriber(
            '/dog_pos', 
            Odometry, 
            self.raw_dog_pos_callback,
            queue_size=10
        )
        
        self.target_sub = rospy.Subscriber(
            '/target_ekf_odom',
            Odometry,
            self.target_callback,
            queue_size=10
        )
        
        self.vins_sub = rospy.Subscriber(
            '/vins_fusion/imu_propagate',
            Odometry,
            self.vins_callback,
            queue_size=10
        )
        # 订阅AOA数据
        self.aoa_sub = rospy.Subscriber(
            '/AOA_Tag_data',
            Odometry,
            self.aoa_callback,
            queue_size=10
        )

        # 订阅光流高度（用于AOA距离勾股修正）
        self.flow_sub = rospy.Subscriber(
            '/flow_data',
            Odometry,
            self.flow_callback,
            queue_size=10
        )
        
        self.takeoff_sub = rospy.Subscriber(
            '/px4ctrl/takeoff_land',
            TakeoffLand,
            self.takeoff_callback,
            queue_size=10
        )
        
        self.yaw_diff_preset_sub = rospy.Subscriber(
            '/yaw_diff_preset',
            Float64,
            self.yaw_diff_callback,
            queue_size=10
        )

        # 测试话题订阅者 - 用于重置initialized状态
        self.test_reset_sub = rospy.Subscriber(
            '/test_reset_initialized',
            Bool,
            self.test_reset_callback,
            queue_size=10
        )
        
        # Trigger订阅者
        self.trigger_sub = rospy.Subscriber(
            '/triger',
            PoseStamped,
            self.trigger_callback,
            queue_size=10
        )
        
        # 定时器
        self.timer = rospy.Timer(rospy.Duration(0.05), self.process_callback)
        self.status_timer = rospy.Timer(rospy.Duration(0.1), self.status_check_callback)  # 100ms检查状态
        
        rospy.loginfo("Dog Position Processor initialized")
        rospy.loginfo("Waiting for takeoff signal and yaw diff from traj_server...")
        rospy.loginfo("Initial yaw offset: 0.0 rad (0.0 deg)")
    
    
    def normalize_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def raw_dog_pos_callback(self, msg):
        """处理原始dog位置数据（完全按照callback.cpp的逻辑）"""
        self.raw_dog_pos = msg
        self.raw_dog_pos_count += 1
        
        # 提取位置、速度、yaw
        current_raw_dog_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        current_raw_dog_yaw = msg.pose.pose.orientation.w
        
        # 速度限制（完全按照callback.cpp）
        min_dog_velocity = -2.0
        max_dog_velocity = 2.0
        current_raw_dog_vel[0] = np.clip(current_raw_dog_vel[0], min_dog_velocity, max_dog_velocity)
        current_raw_dog_vel[1] = np.clip(current_raw_dog_vel[1], min_dog_velocity, max_dog_velocity)
        current_raw_dog_vel[2] = np.clip(current_raw_dog_vel[2], min_dog_velocity, max_dog_velocity)
        
        # 速度滤波（完全按照callback.cpp的逻辑）
        if not self.dog_vel_initialized:
            self.raw_dog_vel = current_raw_dog_vel
            self.raw_dog_yaw = current_raw_dog_yaw
            self.dog_vel_initialized = True
        else:
            # 狗头
            self.raw_dog_vel[0] = 0.7 * self.raw_dog_vel[0] + 0.3 * current_raw_dog_vel[0]
            # 狗侧
            self.raw_dog_vel[1] = 0.7 * self.raw_dog_vel[1] + 0.3 * current_raw_dog_vel[1]
            # 狗上
            self.raw_dog_vel[2] = current_raw_dog_vel[2]
            
            # yaw滤波（完全按照callback.cpp）
            delta_yaw = current_raw_dog_yaw - self.raw_dog_yaw
            delta_yaw = self.normalize_angle(delta_yaw)
            self.raw_dog_yaw += 0.2 * delta_yaw

            self.update_dog_yaw_rate(delta_yaw)
        
        # 收到raw dog pos后立即发布处理后的dog_pos
        self.publish_processed_dog_pos()
    
    def update_dog_yaw_rate(self, delta_yaw):
        """更新狗通信角速度
        
        参数:
            delta_yaw: 角度差（已经过normalize_angle处理）
        """
        current_time = rospy.Time.now().to_sec()
        
        if not self.dog_yaw_rate_initialized:
            # 首次初始化
            self.last_dog_yaw_time = current_time
            self.dog_yaw_rate = 0.0
            self.dog_yaw_rate_initialized = True
        else:
            # 计算时间差
            dt = current_time - self.last_dog_yaw_time
            
            if dt > 0.001:  # 避免除零，至少1ms间隔
                # 计算瞬时角速度（delta_yaw已经处理了角度循环问题）
                instant_yaw_rate = delta_yaw / dt
                
                # 简单滤波
                self.dog_yaw_rate = (1 - self.yaw_rate_filter_gain) * self.dog_yaw_rate + \
                                   self.yaw_rate_filter_gain * instant_yaw_rate
                
                # 更新上一帧时间
                self.last_dog_yaw_time = current_time
    
    def target_callback(self, msg):
        """处理目标数据（来自read模块的target_ekf_odom）"""
        # 从target_ekf_odom中解算yaw（仅使用该消息的四元数）
        self.target_dog_yaw = msg.pose.pose.orientation.w
        
        # 从target_ekf_odom中提取位置
        self.target_dog_pos[0] = msg.pose.pose.position.x
        self.target_dog_pos[1] = msg.pose.pose.position.y
        self.target_dog_pos[2] = msg.pose.pose.position.z
        
        self.target_count += 1

        # 仿真模式：直接在目标回调中发布 dog_pos_processed
        if self.simulate_mode:
            out = Odometry()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = "world"

            # 位置
            out.pose.pose.position.x = msg.pose.pose.position.x
            out.pose.pose.position.y = msg.pose.pose.position.y
            out.pose.pose.position.z = msg.pose.pose.position.z

            # ready flags
            out.pose.pose.orientation.w = 1.0  # initialized_
            out.pose.pose.orientation.x = 0.0
            out.pose.pose.orientation.y = 1.0  # precise_pos_offset_ready_
            out.pose.pose.orientation.z = 1.0  # precise_yaw_offset_ready_

            # 速度
            out.twist.twist.linear.x = msg.twist.twist.linear.x
            out.twist.twist.linear.y = msg.twist.twist.linear.y
            out.twist.twist.linear.z = msg.twist.twist.linear.z

            # yaw 和 yaw_rate
            out.twist.twist.angular.x = self.target_dog_yaw
            out.twist.twist.angular.y = 0.0
            out.twist.twist.angular.z = 0.0

            self.dog_pos_pub.publish(out)

    def vins_callback(self, msg):
        """处理VINS数据"""
        # 从VINS消息中提取yaw
        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        # 计算偏航角
        siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        self.vins_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 提取VINS姿态
        quat = [q_x, q_y, q_z, q_w]  # scipy使用xyzw格式
        rot = Rotation.from_quat(quat)
        self.R_wb = rot.as_matrix()  # 世界系到机体系的旋转矩阵
        
        # 提取VINS位置
        self.vins_pos[0] = msg.pose.pose.position.x
        self.vins_pos[1] = msg.pose.pose.position.y
        self.vins_pos[2] = msg.pose.pose.position.z
        
        # 增加计数器
        self.vins_count += 1
    
    def takeoff_callback(self, msg):
        """处理起飞信号"""
        if msg.takeoff_land_cmd == 1:  # 起飞命令
            rospy.loginfo("Takeoff detected, yaw offset will be reinitialized")
            self.initialized = False  # 重置初始化状态
            self.precise_yaw_offset_ready = False
            self.precise_pos_offset_ready = False
            self.trigger_received = False
            # 重置速度相关变量
            self.dog_yaw_rate_initialized = False
            self.dog_vel_initialized = False
            self.dog_yaw_rate = 0.0

    
    def yaw_diff_callback(self, msg):
        """处理yaw差值信息"""
        # 每次降落的时候，发布并保存yaw差值，用于下次起飞时初始化
        self.saved_yaw_diff = msg.data
        rospy.loginfo(f"Received yaw diff: {math.degrees(msg.data):.1f} deg")
    
    def test_reset_callback(self, msg):
        """测试重置回调 - 接收到任意信息时重置initialized状态"""
        rospy.loginfo("Test reset signal received, resetting initialized to False")
        self.initialized = False
        self.precise_yaw_offset_ready = False
        self.precise_pos_offset_ready = False
    
    def trigger_callback(self, msg):
        """Trigger回调（使用PoseStamped，占位内容无需读取）"""
        self.trigger_received = True
        rospy.loginfo("Trigger received - starting iteration")
    
    def publish_aoa_base_angle_control(self, angle):
        """发布AOA base angle控制命令"""
        msg = Float64()
        msg.data = angle
        self.aoa_base_angle_pub.publish(msg)
        # rospy.loginfo("Published AOA base angle control: %.1f deg" % math.degrees(angle))
    
    def status_check_callback(self, event):
        """状态检查回调（100ms频率）"""
        # 检查dog_pos_received状态（完全按照traj_server.cpp）
        if self.raw_dog_pos_count != self.last_raw_dog_pos_count:
            self.raw_dog_pos_received = True
            self.last_raw_dog_pos_count = self.raw_dog_pos_count
            self.last_dog_pos_timer = 0
        else:
            self.last_dog_pos_timer += 1
            if self.last_dog_pos_timer >= 5:  # 连续5次没有新包才重置
                self.raw_dog_pos_received = False
        
        # 检查target_receive状态（模仿dog_pos_received的逻辑）
        if self.target_count != self.last_target_count:
            self.last_target_timer += 1
            if self.last_target_timer >= 1:
                self.target_receive = True
            self.last_target_count = self.target_count
            self.last_target_loss_timer = 0
        else:
            self.last_target_loss_timer += 1
            if self.last_target_loss_timer >= 1:  # 连续1次没有新包才重置
                self.target_receive = False
            self.last_target_timer = 0
        
        # 检查vins_received状态（模仿dog_pos_received的逻辑）
        if self.vins_count != self.last_vins_count:
            self.vins_received = True
            self.last_vins_count = self.vins_count
            self.last_vins_timer = 0
        else:
            self.last_vins_timer += 1
            if self.last_vins_timer >= 5:  # 连续5次没有新包才重置
                self.vins_received = False

        # 检查aoa_received状态（模仿dog_pos_received的逻辑）
        if self.aoa_count != self.last_aoa_count:
            self.aoa_received = True
            self.last_aoa_count = self.aoa_count
            self.last_aoa_timer = 0
        else:
            self.last_aoa_timer += 1
            if self.last_aoa_timer >= 5:
                self.aoa_received = False

    def process_callback(self, event):
        """主处理回调（完全按照traj_server.cpp的逻辑）"""
        
        # 处理起飞重新初始化
        if (not self.initialized) and self.raw_dog_pos_received and self.vins_received:
            # 使用新的初始化方式：raw_dog_yaw - (vins_yaw + diff)
            if self.saved_yaw_diff is not None:
                diff = self.saved_yaw_diff
                self.saved_yaw_diff = None
            else:
                diff = 0.0  # 默认diff=0
            
            self.yaw_offset = self.normalize_angle(self.raw_dog_yaw - (self.vins_yaw + diff))
            
            # 位置偏移：raw_dog_pos
            # 将vins位置按照yaw_offset旋转后，加到pos_offset后面
            cos_yaw = math.cos(self.yaw_offset)
            sin_yaw = math.sin(self.yaw_offset)
            rotated_vins_x = cos_yaw * self.vins_pos[0] - sin_yaw * self.vins_pos[1]
            rotated_vins_y = sin_yaw * self.vins_pos[0] + cos_yaw * self.vins_pos[1]
            rotated_vins_pos = np.array([rotated_vins_x, rotated_vins_y, self.vins_pos[2]])
            tmp_p = np.array([
                self.raw_dog_pos.pose.pose.position.x,
                self.raw_dog_pos.pose.pose.position.y,
                self.raw_dog_pos.pose.pose.position.z
            ])
            self.pos_offset = tmp_p - rotated_vins_pos
            
            rospy.loginfo(f"Reinitialized offsets: yaw_offset={math.degrees(self.yaw_offset):.1f}deg, "
                         f"pos_offset=[{self.pos_offset[0]:.3f}, {self.pos_offset[1]:.3f}, {self.pos_offset[2]:.3f}]")
            
            # 自动进入预设模式
            self.initialized = True  # 初始化完成
            self.precise_pos_offset_ready = True
            self.precise_yaw_offset_ready = True
        
        # 处理hc14_dog数据：当同时收到target和hc14数据时，维护yaw和pos差值补偿；如果使用预设offset，则不进行迭代
        # 只有在初始化后且收到trigger后才进行offset迭代
        if self.target_receive and self.raw_dog_pos_received and self.vins_received:
            # 正常的yaw offset迭代逻辑
            current_yaw_offset = self.normalize_angle(self.raw_dog_yaw - self.target_dog_yaw)
            yaw_offset_diff = self.normalize_angle(current_yaw_offset - self.yaw_offset)
            # 处理角度环绕问题
            yaw_offset_diff = self.normalize_angle(yaw_offset_diff)
            
            # 对yaw offset进行线性滤波，防止突变
            self.yaw_offset += self.yaw_filter_gain * yaw_offset_diff
            
            # 标记hc14_dog信息可用，采用计数器方式防止抖动
            if abs(yaw_offset_diff) < self.yaw_stable_threshold:
                if not self.precise_yaw_offset_ready:
                    rospy.loginfo("precise_yaw_offset_ready!")
                    if self.precise_pos_offset_ready:
                        self.initialized = True  # 通过迭代达到precise_yaw_offset_ready和precise_pos_offset_ready，设置initialized
                self.precise_yaw_offset_ready = True
            
            if abs(yaw_offset_diff) > self.yaw_exceed_threshold:
                self.yaw_exceed_timer += 1
                if self.yaw_exceed_timer > self.yaw_exceed_max_count:
                    rospy.logwarn("yaw_offset_diff too large!")
                    self.precise_yaw_offset_ready = False
                    self.yaw_exceed_timer = 0
            else:
                self.yaw_exceed_timer = 0

            if self.precise_yaw_offset_ready:
                # 更新位置偏移：raw_dog_pos - vins_pos
                raw_dog_pos = np.array([
                    self.raw_dog_pos.pose.pose.position.x,
                    self.raw_dog_pos.pose.pose.position.y,
                    self.raw_dog_pos.pose.pose.position.z
                ])

                # 计算前馈偏移量（基于final_dog_vel，在世界坐标系下）
                # 参考traj_server: offset = camera_offset * vel
                # target_dog_pos先加上前馈偏移量（在世界坐标系下）
                target_dog_pos_with_ff = np.array([
                    self.target_dog_pos[0] + self.camera_offset * self.final_dog_vel[0],
                    self.target_dog_pos[1] + self.camera_offset * self.final_dog_vel[1],
                    self.target_dog_pos[2]
                ])
                
                # 然后将加上前馈后的target_dog_pos旋转到狗坐标系
                cos_yaw = math.cos(self.yaw_offset)
                sin_yaw = math.sin(self.yaw_offset)
                rotated_target = np.array([
                    cos_yaw * target_dog_pos_with_ff[0] - sin_yaw * target_dog_pos_with_ff[1],
                    sin_yaw * target_dog_pos_with_ff[0] + cos_yaw * target_dog_pos_with_ff[1],
                    target_dog_pos_with_ff[2]
                ])
                
                current_pos_offset = raw_dog_pos - rotated_target
                
                # 对位置偏移进行线性滤波，防止突变
                pos_offset_diff = current_pos_offset - self.pos_offset
                self.pos_offset += self.pos_filter_gain * pos_offset_diff
                
                # 计算位置偏移差值
                pos_offset_diff_norm = np.linalg.norm(current_pos_offset - self.pos_offset)
                
                # 标记位置偏移是否精确ready
                if pos_offset_diff_norm < self.pos_stable_threshold:
                    if not self.precise_pos_offset_ready:
                        rospy.loginfo("precise_pos_offset_ready!")
                        if self.precise_yaw_offset_ready:
                            self.initialized = True  # 通过迭代达到precise_yaw_offset_ready和precise_pos_offset_ready，设置initialized
                        # 位置收敛时重置卡尔曼滤波器
                        rospy.loginfo("Position converged, resetting Kalman filter")
                        self.kf.reset()
                        self.last_kf_time = None
                        self.yaw_filter_initialized = False
                        self.filtered_yaw = 0.0
                    self.precise_pos_offset_ready = True
                
                if pos_offset_diff_norm > self.pos_exceed_threshold:
                    self.pos_exceed_timer += 1
                    if self.pos_exceed_timer > self.pos_exceed_max_count:
                        rospy.logwarn("pos_offset_diff too large!")
                        self.precise_pos_offset_ready = False
                        self.pos_exceed_timer = 0
                else:
                    self.pos_exceed_timer = 0
        
        # 维护AOA位置偏移（单距离单角度方案）
        # 前提：yaw ready、vins可用、aoa可用，且飞机朝向与当前处理后狗位姿夹角在±30度内
        if self.precise_yaw_offset_ready and self.vins_received and self.aoa_received and self.aoa_distance > self.aoa_min_distance:
            # 角度限制：飞机朝向需面向当前processed dog pos的±30度
            dog_vec = self.final_dog_pos - self.vins_pos
            heading_to_dog = math.atan2(dog_vec[1], dog_vec[0])
            heading_diff = self.normalize_angle(heading_to_dog - self.vins_yaw)
            if abs(heading_diff) > math.pi / 3.0:  # 60度
                return
        
            height_diff = self.final_dog_pos[2] - self.vins_pos[2]
            if self.aoa_distance < abs(height_diff):
                return
            aoa_distance_horizontal = math.sqrt(self.aoa_distance * self.aoa_distance - height_diff * height_diff)

            # ========================== 坐标与几何问题定义 ==========================
            #
            # 目标：
            #   已知：
            #     - 飞机在世界系中的姿态 R_wb
            #     - 飞机为原点
            #     - 目标相对飞机的高度差 height_diff = Δz
            #     - 目标相对飞机的水平距离 aoa_distance_horizontal
            #     - 目标在机体系下的 yaw 角 aoa_angle
            #
            #   求：
            #     - 目标在世界系下的位置 aoa_pos_world
            #
            # 约束条件：
            #   1) 目标位于"yaw 平面"内（由 yaw 角确定）
            #   2) 目标的 z 坐标等于 Δz
            #   3) 目标的水平距离等于 aoa_distance_horizontal
            #
            # 几何思路：
            #   - yaw 平面 ∩ 水平面 (z = Δz) 是一条直线
            #   - 先在这条直线上找到"距离原点最近的点 p0"
            #   - 再沿该直线方向走到满足水平距离约束的位置
            # ======================================================================

            # ========================== Step 0：基础向量 ==========================

            # 世界系 z 轴单位向量
            ez = np.array([0.0, 0.0, 1.0])

            # 目标在机体系 yaw 平面中的 bearing 方向（x-y 平面内）
            bearing_b = np.array([
                math.cos(self.aoa_angle),
                math.sin(self.aoa_angle),
                0.0
            ])

            # ========================== Step 1：构造 yaw 平面的法向 ==========================
            #
            # yaw 平面定义：
            #   - 经过飞机原点
            #   - 包含 bearing_b 方向
            #   - 包含机体系 z 轴
            #
            # 因此其法向（机体系）为：
            n_b = np.cross(bearing_b, ez)

            # 转到世界系
            n_w = self.R_wb @ n_b

            # 归一化，便于后续解释和数值稳定
            n_w_norm = np.linalg.norm(n_w)
            if n_w_norm < 1e-8:
                return
            n_w = n_w / n_w_norm

            # ========================== Step 2：yaw 平面与水平面的交线 ==========================
            #
            # yaw 平面与水平面 (z = const) 的交集是一条直线
            # 其方向等于两个平面法向的叉乘：
            #   d ∥ n × ez
            #
            d_w = - np.cross(n_w, ez)

            # 退化情况：yaw 平面与水平面平行（无唯一解）
            if np.linalg.norm(d_w) < 1e-6:
                # 此时仅凭水平距离无法唯一确定目标
                return

            # 该交线的单位方向（用于后续参数化）
            d_hat = d_w / np.linalg.norm(d_w)

            # |n × ez|，表示 yaw 平面对水平面的倾斜程度
            d_norm = np.linalg.norm(d_w)

            # ========================== Step 3：构造"最近点" p0 ==========================
            #
            # 目标：
            #   在 yaw 平面 ∩ z = Δz 的直线上，
            #   找到距离原点最近的点 p0。
            #
            # 思路：
            #   - 从 p = Δz * ez 出发（只满足高度约束）
            #   - 只能在"水平面内"修正，否则会破坏 z = Δz
            #   - 修正方向必须取 yaw 平面法向 n 在水平面的投影
            #

            # yaw 平面法向在竖直方向的分量
            n_z = np.dot(n_w, ez)

            # yaw 平面法向在水平面的分量（正确的"拉回方向"）
            n_h = n_w - n_z * ez

            # |n_h|^2
            n_h_sq = np.dot(n_h, n_h)

            # 理论上此处不应退化（与 d_w 检查等价），保险起见
            if n_h_sq < 1e-8:
                return

            # ========================== Step 4：一步写出 p0（闭式解） ==========================
            #
            # 构造未知点：
            #   p0 = Δz * ez - α * n_h
            #
            # 其中 α 表示沿 n_h 方向修正的量。
            # 将 p0 代入 yaw 平面方程：
            #   n · p0 = 0
            #
            # 得到一元一次方程：
            #   Δz * n_z - α * |n_h|^2 = 0
            #
            # 解得：
            #   α = (n_z * Δz) / |n_h|^2
            #
            p0 = height_diff * ez - (n_z * height_diff / n_h_sq) * n_h

            # ========================== Step 5：在 yaw 平面内施加水平距离约束 ==========================
            #
            # yaw 平面 ∩ z = Δz 是一条直线：
            #   L(t) = p0 + t * d_hat
            #
            # 其中：
            #   - p0     是该直线中距离原点最近的点
            #   - d_hat  是直线的单位方向（完全在水平面内）
            #
            # 目标点必须满足：
            #   || (p0 + t * d_hat)_h || = aoa_distance_horizontal
            #

            # p0 的水平分量
            p0_h = np.array([p0[0], p0[1]])

            # 最近点到原点的水平距离
            p0_h_norm = np.linalg.norm(p0_h)

            # 解二次方程前的判定
            inside = aoa_distance_horizontal * aoa_distance_horizontal - p0_h_norm * p0_h_norm

            if inside < 0.0:
                # 水平圆与 yaw 平面直线无交点
                return

            # 沿 bearing 正方向的唯一物理解
            t = math.sqrt(inside)

            # 最终目标相对飞机的位置（世界系）
            target_rel = p0 + t * d_hat

            # ========================== Step 6：转为世界系绝对坐标 ==========================
            #
            # 前面所有计算均以飞机为原点
            # 最终只在这里加上飞机世界系位置
            #
            aoa_pos_world = self.vins_pos + target_rel

            # 将AOA估计位置旋转到狗坐标系（使用 yaw_offset）
            cos_yaw = math.cos(self.yaw_offset)
            sin_yaw = math.sin(self.yaw_offset)
            rotated_aoa_x = cos_yaw * (aoa_pos_world[0] - self.vins_pos[0]) - sin_yaw * (aoa_pos_world[1] - self.vins_pos[1])
            rotated_aoa_y = sin_yaw * (aoa_pos_world[0] - self.vins_pos[0]) + cos_yaw * (aoa_pos_world[1] - self.vins_pos[1])
            rotated_aoa_pos = np.array([rotated_aoa_x, rotated_aoa_y, self.final_dog_pos[2]])

            # 与raw_dog_pos比较，更新pos_offset
            raw_dog_pos = np.array([
                self.raw_dog_pos.pose.pose.position.x,
                self.raw_dog_pos.pose.pose.position.y,
                self.raw_dog_pos.pose.pose.position.z
            ])
            current_pos_offset = raw_dog_pos - rotated_aoa_pos

            # Debug发布AOA估计位置
            aoa_debug_msg = Odometry()
            aoa_debug_msg.header.stamp = rospy.Time.now()
            aoa_debug_msg.header.frame_id = "world"
            aoa_debug_msg.pose.pose.position.x = aoa_pos_world[0]
            aoa_debug_msg.pose.pose.position.y = aoa_pos_world[1]
            aoa_debug_msg.pose.pose.position.z = aoa_pos_world[2]
            self.aoa_dog_pos_debug_pub.publish(aoa_debug_msg)

            # 位置偏移滤波：x和y分开迭代，z不迭代
            pos_offset_diff = current_pos_offset - self.pos_offset
            
            # x方向迭代
            diff_x = pos_offset_diff[0]
            if abs(diff_x) > 1e-6:
                step_x = self.aoa_pos_filter_gain * abs(diff_x)
                step_x = min(step_x, self.aoa_pos_step_limit)
                self.pos_offset[0] += step_x if diff_x > 0 else -step_x
            
            # y方向迭代
            diff_y = pos_offset_diff[1]
            if abs(diff_y) > 1e-6:
                step_y = self.aoa_pos_filter_gain * abs(diff_y)
                step_y = min(step_y, self.aoa_pos_step_limit)
                self.pos_offset[1] += step_y if diff_y > 0 else -step_y
            
            # z方向不迭代，保持不变


    def publish_processed_dog_pos(self):
        """发布处理后的dog_pos（使用Odometry格式）"""
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        # 位置处理：先减掉pos_offset，再旋转yaw_offset
        raw_pos = np.array([
            self.raw_dog_pos.pose.pose.position.x,
            self.raw_dog_pos.pose.pose.position.y,
            self.raw_dog_pos.pose.pose.position.z
        ])
        
        # 先减掉pos_offset（狗坐标系下的偏移）
        corrected_pos = raw_pos - self.pos_offset
        
        # 再旋转yaw_offset（将狗坐标系转换到世界坐标系）
        yaw_diff = self.normalize_angle(- self.yaw_offset)
        cos_yaw = math.cos(yaw_diff)
        sin_yaw = math.sin(yaw_diff)
        rotated_x = cos_yaw * corrected_pos[0] - sin_yaw * corrected_pos[1]
        rotated_y = sin_yaw * corrected_pos[0] + cos_yaw * corrected_pos[1]
        
        # 速度旋转
        target_yaw = self.normalize_angle(self.raw_dog_yaw - self.yaw_offset)
        cos_yaw = math.cos(target_yaw)
        sin_yaw = math.sin(target_yaw)
        rotated_vx = cos_yaw * self.raw_dog_vel[0] - sin_yaw * self.raw_dog_vel[1]
        rotated_vy = sin_yaw * self.raw_dog_vel[0] + cos_yaw * self.raw_dog_vel[1]

        


        # 先计算并存储所有最终输出的dog相关量到self中
        self.final_dog_pos = np.array([
            rotated_x,
            rotated_y,
            corrected_pos[2]
        ])

        self.final_dog_vel = np.array([
            rotated_vx,
            rotated_vy,
            self.raw_dog_vel[2]
        ])
        self.final_dog_yaw = target_yaw

        # 如果target_receive为true，直接用target_dog_pos替换位置（在KF之前）
        # 加上camera_offset前馈偏移量（基于final_dog_vel，在世界坐标系下）
        if self.target_receive:
            self.final_dog_pos[0] = self.target_dog_pos[0] + self.camera_offset * self.final_dog_vel[0]
            self.final_dog_pos[1] = self.target_dog_pos[1] + self.camera_offset * self.final_dog_vel[1]
            self.final_dog_pos[2] = self.target_dog_pos[2]
            self.final_dog_yaw = self.target_dog_yaw

        # 卡尔曼滤波：只有pos收敛了才使用滤波器
        if self.kf_enabled and self.precise_pos_offset_ready:
            current_time = rospy.Time.now().to_sec()
            
            if self.last_kf_time is None:
                self.last_kf_time = current_time
                dt = 0.05  # 默认50ms
            else:
                dt = current_time - self.last_kf_time
                if dt <= 0:
                    dt = 0.05  # 防止非正时间间隔
                
                # 如果dt超过超时阈值，重置滤波器
                if dt > self.kf_timeout:
                    rospy.logwarn(f"Kalman filter timeout (dt={dt:.2f}s > {self.kf_timeout:.2f}s), resetting filter")
                    self.kf.reset()
                    self.yaw_filter_initialized = False
                    self.filtered_yaw = 0.0
                    dt = 0.05  # 重置后使用默认时间间隔
            
            # 对位置和速度进行卡尔曼滤波
            filtered_pos, filtered_vel = self.kf.filter(
                self.final_dog_pos.copy(),
                self.final_dog_vel.copy(),
                dt
            )
            
            # 对 yaw 进行简单一阶滤波（处理角度环绕）
            if not self.yaw_filter_initialized:
                self.filtered_yaw = self.final_dog_yaw
                self.yaw_filter_initialized = True
            else:
                yaw_diff = self.normalize_angle(self.final_dog_yaw - self.filtered_yaw)
                self.filtered_yaw = self.normalize_angle(
                    self.filtered_yaw + self.yaw_filter_gain_kf * yaw_diff
                )
            
            # 使用滤波后的值
            self.final_dog_pos = filtered_pos
            self.final_dog_vel = filtered_vel
            self.final_dog_yaw = self.filtered_yaw
            
            self.last_kf_time = current_time

        # 使用self的变量填充msg
        msg.pose.pose.position.x = self.final_dog_pos[0]
        msg.pose.pose.position.y = self.final_dog_pos[1]
        msg.pose.pose.position.z = self.final_dog_pos[2]

        # 状态位：w=initialized, x=aoa_converged, y=precise_pos_ready, z=precise_yaw_ready
        msg.pose.pose.orientation.w = 1.0 if self.initialized else 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 1.0 if self.precise_pos_offset_ready else 0.0
        msg.pose.pose.orientation.z = 1.0 if self.precise_yaw_offset_ready else 0.0

        # 速度
        msg.twist.twist.linear.x = self.final_dog_vel[0]
        msg.twist.twist.linear.y = self.final_dog_vel[1]
        msg.twist.twist.linear.z = self.final_dog_vel[2]
        
        # 最终yaw放在twist.angular.x,另外，y和z放上转换后的狗的速度
        msg.twist.twist.angular.x = self.final_dog_yaw
        msg.twist.twist.angular.y = self.dog_yaw_rate  # 发布狗通信角速度
        msg.twist.twist.angular.z = 0.0

        self.dog_pos_pub.publish(msg)

    def aoa_callback(self, msg):
        # 单距离 + 单角度：position.x = 距离，orientation.x = 相对于无人机朝向的角度
        distance_raw = msg.pose.pose.position.x
        angle_raw = msg.pose.pose.orientation.x

        self.aoa_distance = distance_raw
        self.aoa_angle = angle_raw

        self.aoa_count += 1

    def flow_callback(self, msg):
        # 光流高度在 position.z（参考 withdraw 的 /flow_data 使用 Odometry）
        self.flow_z = msg.pose.pose.position.z
    
    
def main():
    try:
        processor = DogPosProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dog Position Processor shutdown")
    except Exception as e:
        rospy.logerr(f"Dog Position Processor error: {e}")

if __name__ == '__main__':
    main()
