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

class DogPosProcessor:
    def __init__(self):
        rospy.init_node('dog_pos_processor', anonymous=True)
        
        self.aoa_base_angle = math.radians(0.0)
        
        # AOA base angle控制相关变量
        self.aoa_base_angle_auto = False  # 是否自动计算aoa_base_angle
        self.aoa_rotation_speed = math.radians(10.0)  # 旋转速度（度/秒）
        self.aoa_rotation_timer = 0.0
        self.aoa_last_received_time = 0.0
        self.aoa_rotation_started = False
        self.aoa_yaw_target_reached = False
        # 手动覆盖：开启后强制使用手动角度，不自动调整、不旋转
        self.aoa_manual_override = True
        self.aoa_base_angle_manual = 0.0
        
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

        # AOA相关参数
        self.aoa_min_distance = 0.5  # m
        self.aoa_max_angle_deg = 80.0  # deg
        self.aoa_filter_gain = 0.8  # 位置融合增益（趋近AOA）
        self.aoa_alpha_yaw = 0.000   # yaw融合增益
        self.aoa_pos_thresh = 0.05  # m，AOA收敛阈值
        self.aoa_yaw_thresh = math.radians(5.0)  # rad
        
        # 最终输出的dog相关量存储
        self.final_dog_pos = np.array([0.0, 0.0, 0.0])  # 最终输出的dog位置
        self.final_dog_vel = np.array([0.0, 0.0, 0.0])  # 最终输出的dog速度
        self.final_dog_yaw = 0.0  # 最终输出的dog yaw
        self.final_pos_wo_aoa = np.array([0.0, 0.0, 0.0])  # 不含AOA调整的最终位置
        self.final_yaw_wo_aoa = 0.0  # 不含AOA调整的最终yaw
        
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
        self.aoa_distance = 0.0
        self.aoa_angle = 0.0
        self.aoa_delta_p = np.array([0.0, 0.0, 0.0])
        self.aoa_delta_yaw = 0.0
        self.aoa_converged = False
        self.aoa_received = False
        self.aoa_count = 0
        self.last_aoa_count = 0
        self.last_aoa_timer = 0

        # 光流高度（用于修正AOA距离的垂直分量）
        self.flow_z = 0.0
        self.flow_height_bias = 0.47  # 与withdraw一致的零偏
        
        
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
        
        # 处理后的数据
        self.final_pos_wo_aoa = np.array([0.0, 0.0, 0.0])
        self.final_yaw_wo_aoa = 0.0

        # 初始化标志
        self.dog_vel_initialized = False
        
        # 发布者 - 使用Odometry格式
        self.dog_pos_pub = rospy.Publisher(
            '/dog_pos_processed',
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
        
        # AOA base angle控制话题发布者
        self.aoa_base_angle_pub = rospy.Publisher(
            '/aoa_base_angle_control',
            Float64,
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
        
        # 收到raw dog pos后立即发布处理后的dog_pos
        self.publish_processed_dog_pos()
    
    def target_callback(self, msg):
        """处理目标数据（来自read模块的target_ekf_odom）"""
        # 从target_ekf_odom中提取yaw（假设在orientation.w中）
        self.target_dog_yaw = msg.pose.pose.orientation.w
        
        # 从target_ekf_odom中提取位置
        self.target_dog_pos[0] = msg.pose.pose.position.x
        self.target_dog_pos[1] = msg.pose.pose.position.y
        self.target_dog_pos[2] = msg.pose.pose.position.z
        
        self.target_count += 1

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

            # 重置AOA相关增量
            self.aoa_delta_p[:] = 0.0
            self.aoa_delta_yaw = 0.0
            self.aoa_converged = False
    
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
        
        # 同步清空AOA状态
        self.aoa_delta_p[:] = 0.0
        self.aoa_delta_yaw = 0.0
        self.aoa_converged = False
    
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
            self.target_receive = True
            self.last_target_count = self.target_count
            self.last_target_timer = 0
        else:
            self.last_target_timer += 1
            if self.last_target_timer >= 5:  # 连续5次没有新包才重置
                self.target_receive = False
        
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
            
            self.yaw_offset = self.raw_dog_yaw - (self.vins_yaw + diff)
            
            # 位置偏移：raw_dog_pos
            # 将vins位置按照-yaw_offset旋转后，加到pos_offset后面
            cos_neg_yaw = math.cos(self.yaw_offset)
            sin_neg_yaw = math.sin(self.yaw_offset)
            rotated_vins_x = cos_neg_yaw * self.vins_pos[0] - sin_neg_yaw * self.vins_pos[1]
            rotated_vins_y = sin_neg_yaw * self.vins_pos[0] + cos_neg_yaw * self.vins_pos[1]
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
        
        # 处理hc14_dog数据：当同时收到target和hc14数据时，维护yaw和pos差值补偿；如果使用预设offset，则不进行迭代
        # 只有在初始化后且收到trigger后才进行offset迭代
        if self.trigger_received and self.target_receive and self.raw_dog_pos_received:
            # 正常的yaw offset迭代逻辑
            current_yaw_offset = self.raw_dog_yaw - self.target_dog_yaw
            yaw_offset_diff = current_yaw_offset - self.yaw_offset
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

                # 将世界坐标系下的位置差旋转yaw_offset，得到狗坐标系下的pos_offset
                cos_neg_yaw = math.cos(self.yaw_offset)
                sin_neg_yaw = math.sin(self.yaw_offset)
                
                rotated_x = cos_neg_yaw * self.target_dog_pos[0] - sin_neg_yaw * self.target_dog_pos[1]
                rotated_y = sin_neg_yaw * self.target_dog_pos[0] + cos_neg_yaw * self.target_dog_pos[1]
                
                current_pos_offset = raw_dog_pos - np.array([rotated_x, rotated_y, self.target_dog_pos[2]])
                
                # 对位置偏移进行线性滤波，防止突变
                pos_offset_diff = current_pos_offset - self.pos_offset
                self.pos_offset[0] += self.pos_filter_gain * pos_offset_diff[0]
                self.pos_offset[1] += self.pos_filter_gain * pos_offset_diff[1]
                self.pos_offset[2] += self.pos_filter_gain * pos_offset_diff[2]
                
                # 计算位置偏移差值
                pos_offset_diff = np.linalg.norm(current_pos_offset - self.pos_offset)
                
                # 标记位置偏移是否精确ready
                if pos_offset_diff < self.pos_stable_threshold:
                    if not self.precise_pos_offset_ready:
                        rospy.loginfo("precise_pos_offset_ready!")
                        if self.precise_yaw_offset_ready:
                            self.initialized = True  # 通过迭代达到precise_yaw_offset_ready和precise_pos_offset_ready，设置initialized
                    self.precise_pos_offset_ready = True
                
                if pos_offset_diff > self.pos_exceed_threshold:
                    self.pos_exceed_timer += 1
                    if self.pos_exceed_timer > self.pos_exceed_max_count:
                        rospy.logwarn("pos_offset_diff too large!")
                        self.precise_pos_offset_ready = False
                        self.pos_exceed_timer = 0
                else:
                    self.pos_exceed_timer = 0
        
        # 维护AOA增量（在具备vins与原始狗位姿时）
        # 只有在初始化后且收到trigger后才进行AOA迭代
        if self.trigger_received and self.vins_received and self.aoa_received:
            if self.AOA_valid():
                # 基于AOA的世界系目标点（参考withdraw思路）
                aoa_px = self.vins_pos[0] + self.aoa_distance * math.cos(self.final_yaw_wo_aoa + self.aoa_base_angle + self.aoa_angle)
                aoa_py = self.vins_pos[1] + self.aoa_distance * math.sin(self.final_yaw_wo_aoa + self.aoa_base_angle + self.aoa_angle)
                current_aoa_delta_p = np.array([self.final_pos_wo_aoa[0] - aoa_px, self.final_pos_wo_aoa[1] - aoa_py, 0.0])
                self.aoa_delta_p += self.aoa_filter_gain * (current_aoa_delta_p - self.aoa_delta_p)

                # yaw增量：根据累计aoa_delta_p方向与当前最终yaw的差异做微调
                if np.linalg.norm(self.aoa_delta_p[:2]) > 1e-3:
                    heading_aoa = math.atan2(self.aoa_delta_p[1], self.aoa_delta_p[0])  # AOA累计偏移方向
                    yaw_error = self.normalize_angle(heading_aoa - self.final_yaw_wo_aoa - self.aoa_base_angle)
                    self.aoa_delta_yaw += self.aoa_alpha_yaw * yaw_error

                # 收敛判据：若基础精确已ready则对应增量视为0
                aoa_delta_p_diff = np.linalg.norm(current_aoa_delta_p - self.aoa_delta_p)

                self.aoa_converged = (aoa_delta_p_diff < self.aoa_pos_thresh) or self.precise_pos_offset_ready
            else:
                self.aoa_converged = False
        
        # AOA base angle自动控制逻辑
        self.update_aoa_base_angle()

    
    def publish_processed_dog_pos(self):
        """发布处理后的dog_pos（使用Odometry格式）"""
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        if self.target_receive:
            aoa_dy_effective = 0.0
            aoa_dp_effective = np.array([0.0, 0.0, 0.0])
        else:
            aoa_dy_effective = self.aoa_delta_yaw
            aoa_dp_effective = self.aoa_delta_p.copy()

        # 位置处理：先减掉pos_offset，再旋转yaw_offset
        raw_pos = np.array([
            self.raw_dog_pos.pose.pose.position.x,
            self.raw_dog_pos.pose.pose.position.y,
            self.raw_dog_pos.pose.pose.position.z
        ])
        
        # 先减掉pos_offset（狗坐标系下的偏移）
        corrected_pos = raw_pos - self.pos_offset
        
        # 再旋转yaw_offset（将狗坐标系转换到世界坐标系）
        yaw_diff = self.normalize_angle(- self.yaw_offset - aoa_dy_effective)
        cos_yaw = math.cos(yaw_diff)
        sin_yaw = math.sin(yaw_diff)
        rotated_x = cos_yaw * corrected_pos[0] - sin_yaw * corrected_pos[1]
        rotated_y = sin_yaw * corrected_pos[0] + cos_yaw * corrected_pos[1]
        
        # 速度旋转
        target_yaw = self.normalize_angle(self.raw_dog_yaw - self.yaw_offset - aoa_dy_effective)
        cos_yaw = math.cos(target_yaw)
        sin_yaw = math.sin(target_yaw)
        rotated_vx = cos_yaw * self.raw_dog_vel[0] - sin_yaw * self.raw_dog_vel[1]
        rotated_vy = sin_yaw * self.raw_dog_vel[0] + cos_yaw * self.raw_dog_vel[1]

        


        # 先计算并存储所有最终输出的dog相关量到self中
        self.final_pos_wo_aoa = np.array([
            rotated_x,
            rotated_y,
            corrected_pos[2]
        ])
        self.final_yaw_wo_aoa = self.normalize_angle(self.raw_dog_yaw - self.yaw_offset)
        
        self.final_dog_pos = np.array([
            rotated_x - aoa_dp_effective[0],
            rotated_y - aoa_dp_effective[1],
            corrected_pos[2]
        ])
        self.final_dog_vel = np.array([
            rotated_vx,
            rotated_vy,
            self.raw_dog_vel[2]
        ])
        self.final_dog_yaw = target_yaw

        # 使用self的变量填充msg
        msg.pose.pose.position.x = self.final_dog_pos[0]
        msg.pose.pose.position.y = self.final_dog_pos[1]
        msg.pose.pose.position.z = self.final_dog_pos[2]

        # 状态位：w=initialized, x=aoa_converged, y=precise_pos_ready, z=precise_yaw_ready
        msg.pose.pose.orientation.w = 1.0 if self.initialized else 0.0
        msg.pose.pose.orientation.x = 1.0 if self.aoa_converged else 0.0
        msg.pose.pose.orientation.y = 1.0 if self.precise_pos_offset_ready else 0.0
        msg.pose.pose.orientation.z = 1.0 if self.precise_yaw_offset_ready else 0.0

        # 速度
        msg.twist.twist.linear.x = self.final_dog_vel[0]
        msg.twist.twist.linear.y = self.final_dog_vel[1]
        msg.twist.twist.linear.z = self.final_dog_vel[2]
        
        # 最终yaw放在twist.angular.x,另外，y和z放上转换后的狗的速度
        msg.twist.twist.angular.x = self.final_dog_yaw
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = 0.0

        self.dog_pos_pub.publish(msg)

    def AOA_valid(self):
        if self.aoa_distance is None:
            return False
        # 使用平面距离进行阈值判断
        if self.aoa_distance < self.aoa_min_distance:
            return False
        if abs(math.degrees(self.aoa_angle)) > self.aoa_max_angle_deg:
            return False
        return True

    def aoa_callback(self, msg):
        # AOA距离（position.x）与角度（orientation.w），与仓库约定一致
        self.aoa_angle = msg.pose.pose.orientation.w
        # 用光流高度修正：水平距离 = sqrt(max(0, d^2 - h^2))，直接覆盖 aoa_distance
        height = (self.flow_z - self.flow_height_bias)
        d2 = max(0.0, msg.pose.pose.position.x * msg.pose.pose.position.x - height * height)
        self.aoa_distance = math.sqrt(d2)
        self.aoa_count += 1
        
        # 更新AOA接收时间
        self.aoa_last_received_time = rospy.Time.now().to_sec()
        
        # 检查AOA yaw是否接近0
        self.aoa_yaw_target_reached = abs(self.aoa_angle) < math.radians(5.0)

    def flow_callback(self, msg):
        # 光流高度在 position.z（参考 withdraw 的 /flow_data 使用 Odometry）
        self.flow_z = msg.pose.pose.position.z
    
    def update_aoa_base_angle(self):
        """更新AOA base angle的自动控制逻辑"""
        current_time = rospy.Time.now().to_sec()
        
        # 手动覆盖：强制固定为手动值并直接返回
        if self.aoa_manual_override:
            self.aoa_base_angle = self.aoa_base_angle_manual
            self.aoa_rotation_started = False
            return
        
        if self.aoa_base_angle_auto:
            # 自动模式：计算基于位置差的aoa_base_angle
            if self.vins_received and self.raw_dog_pos_received:
                # 计算final_pos_wo_aoa和vins_pos的差向量
                pos_diff = self.final_pos_wo_aoa - self.vins_pos
                if np.linalg.norm(pos_diff[:2]) > 0.1:  # 避免除零
                    # 计算差向量的yaw角
                    pos_diff_yaw = math.atan2(pos_diff[1], pos_diff[0])
                    # 计算aoa_base_angle = pos_diff_yaw - 180度 - final_yaw_wo_aoa
                    calculated_angle = self.normalize_angle(pos_diff_yaw - math.pi - self.final_yaw_wo_aoa)
                    self.aoa_base_angle = calculated_angle
                    # 发布计算出的角度
                    self.publish_aoa_base_angle_control(calculated_angle)
        
        # 检查是否需要旋转
        if not self.aoa_received or (current_time - self.aoa_last_received_time) > 2.0:
            # 没有AOA数据或超过2秒没收到，开始旋转
            if not self.aoa_rotation_started:
                self.aoa_rotation_started = True
                self.aoa_rotation_timer = 0.0
                rospy.loginfo("AOA rotation started - no AOA data received")
            
            # 缓慢旋转
            self.aoa_rotation_timer += 0.05  # 50ms timer
            rotation_angle = self.aoa_rotation_timer * self.aoa_rotation_speed
            self.aoa_base_angle = self.normalize_angle(rotation_angle)
            # 发布旋转角度
            self.publish_aoa_base_angle_control(self.aoa_base_angle)
            
        elif self.aoa_received and self.aoa_rotation_started:
            # 收到AOA数据，继续旋转直到yaw接近0
            if not self.aoa_yaw_target_reached:
                self.aoa_rotation_timer += 0.05
                rotation_angle = self.aoa_rotation_timer * self.aoa_rotation_speed
                self.aoa_base_angle = self.normalize_angle(rotation_angle)
                # 发布旋转角度
                self.publish_aoa_base_angle_control(self.aoa_base_angle)
            else:
                # AOA yaw接近0，停止旋转
                self.aoa_rotation_started = False
                rospy.loginfo("AOA rotation stopped - yaw target reached")
        
        # 如果AOA收敛，恢复自动模式
        if self.aoa_converged and not self.aoa_base_angle_auto:
            self.aoa_base_angle_auto = True
            rospy.loginfo("AOA converged - switching back to auto mode")

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
