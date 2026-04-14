#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞机速度偏移测试脚本
功能：让飞机在指定时间内匀加速到指定速度，然后保持匀速飞行
支持三种模式：横向飞行、竖向飞行、原地旋转

使用方法：
1. 修改脚本开头的配置参数
2. 运行脚本：python3 test_velocity_offset.py
3. 数据将自动保存到speed_offset_test_*文件夹中

配置参数说明：
- flight_mode: 飞行模式 ("horizontal", "vertical", "yaw_rotation", "z_offset")
- motion_mode: 运动模式 ("variable_speed" 变速运动, "constant_speed" 匀速运动)
- direction: 运动方向 (1 正方向, -1 负方向)
- acceleration_time: 加速时间（秒，仅变速模式使用）
- constant_speed_time: 匀速时间（秒）
- max_velocity: 最大速度（m/s）
- z_height: Z轴高度（米，默认1.0米）
- initial_height: 初始飞行高度（米，默认1.0米）
- z_min: Z轴最小高度（米，默认0.3米，防止撞地）
- z_max: Z轴最大高度（米，默认3.0米，安全上限）
- height_hold_time: 到达初始高度后保持时间（秒，默认3.0秒）
- control_frequency: 控制频率（Hz）
- start_delay: 开始前等待时间（秒）
- final_position_hold_time: 完成后保持位置时间（秒）
- infinite_round_trip: 是否启用无限往返模式（True: 无限往返不停下, False: 单次运动）
- verbose_output: 是否显示详细进度
- yaw_rotation_angle: 旋转角度（度，仅旋转模式）

飞行模式说明：
- horizontal: 横向飞行（X轴运动，Z轴为设定高度）
- vertical: 竖向飞行（Z轴运动）
- yaw_rotation: 原地旋转（保持位置，Yaw角旋转）
- z_offset: Z轴偏移测试（在设定高度进行横向运动）

运动模式说明：
- variable_speed: 变速运动（先匀加速到最大速度，然后保持匀速）
- constant_speed: 匀速运动（从开始就以最大速度匀速运动）

测试流程：
1. 等待指定延迟时间
2. 等待接收vins_fusion位置信息
3. 第一阶段：飞到指定高度并保持一段时间（从当前位置开始）
4. 第二阶段：执行速度偏移测试（根据motion_mode选择变速或匀速）
   - 无限往返模式：从起点到终点，再从终点回到起点，如此循环（按Ctrl+C停止）
   - 单次模式：从起点到终点，然后保持最终位置
5. 保持最终位置（仅单次模式）

设计特点：
- 统一的状态计算函数，同时返回位置、速度、yaw值
- Z轴高度自动限制在安全范围内（z_min ~ z_max）
- 所有飞行模式都遵循相同的安全限制

注意：
- 脚本只限制速度，不限制位置范围，请确保在安全环境下测试
- 初始位置通过订阅/vins_fusion/imu_propagate话题自动获取
- 如果无法获取位置信息，将使用默认位置(0, 0, initial_height)
- Z轴高度有安全限制：最小0.3米，最大3.0米，任何情况下都会遵循这些限制

安全限制：
- Z轴高度自动限制在 [z_min, z_max] 范围内
- 任何情况下都不会超出安全高度范围
"""

import rospy
import numpy as np
import time
import os
from datetime import datetime
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

class VelocityOffsetTest:
    def __init__(self):
        # ==================== 可调参数配置 ====================
        # 飞行模式选择: "horizontal" (横向飞行), "vertical" (竖向飞行), "yaw_rotation" (原地旋转), "z_offset" (Z轴偏移测试)
        self.flight_mode = "horizontal"
        
        # 运动模式选择: "variable_speed" (变速运动), "constant_speed" (匀速运动)
        self.motion_mode = "constant_speed"  # "variable_speed" 或 "constant_speed"
        
        # 运动方向控制: 1 (正方向), -1 (负方向)
        self.direction = -1  # 1 为正方向，-1 为负方向
        
        # 时间参数（单位：秒）
        self.acceleration_time = 2.0      # 加速时间（仅变速模式使用）
        self.constant_speed_time = 5.0    # 匀速时间
        
        # 速度参数（单位：m/s）
        self.max_velocity = 1.0           # 最大速度
        
        # 高度参数（单位：米）
        self.z_height = 1.5               # Z轴高度（默认1米）
        
        # Z轴安全限制（单位：米）
        self.z_min = 0.3                  # Z轴最小高度（防止撞地）
        self.z_max = 3.0                  # Z轴最大高度（安全上限）
        
        # 控制参数
        self.control_frequency = 100        # 控制频率（Hz）
        
        # 测试延迟
        self.start_delay = 2.0            # 开始测试前的等待时间（秒）
        
        # 初始高度设置
        self.initial_height = 1.0         # 初始飞行高度（米）
        self.height_hold_time = 5.0       # 到达初始高度后保持时间（秒）
        
        # 位置保持时间
        self.final_position_hold_time = 2.0  # 测试完成后保持最终位置的时间（秒）
        
        # 无限往返模式
        self.infinite_round_trip = True  # 是否启用无限往返模式（True: 无限往返, False: 单次运动）
        
        # 是否显示详细进度信息
        self.verbose_output = True
        
        # 旋转模式特殊参数
        self.yaw_rotation_angle = 720     # 旋转角度（度），原地旋转模式使用
        
        # 计算总时间
        self.total_time = self.acceleration_time + self.constant_speed_time
        
        # 初始化ROS节点
        rospy.init_node('velocity_offset_test', anonymous=True)
        
        # 创建发布者
        self.pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=10)
        
        # 订阅vins_fusion话题获取当前位置
        self.vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_callback)
        self.current_position = Point(0, 0, 0)  # 当前位置
        self.current_yaw = 0.0  # 当前yaw角
        self.position_received = False  # 是否已接收到位置信息
        self.start_x = 0
        self.start_y = 0
        self.start_yaw = 0

        # 测试状态
        self.start_time = None
        self.test_completed = False

        assert self.flight_mode in ["horizontal", "vertical", "yaw_rotation", "z_offset"]
        assert self.motion_mode in ["variable_speed", "constant_speed"]
        assert self.acceleration_time > 0
        assert self.constant_speed_time > 0
        assert self.max_velocity > 0
        assert self.z_height > 0
        assert self.initial_height > 0
        assert self.height_hold_time > 0
        assert self.final_position_hold_time > 0
        assert self.control_frequency > 0
        assert self.start_delay >= 0
        assert self.yaw_rotation_angle > 0
        assert self.total_time > 0
        assert self.z_min < self.z_max
        assert self.z_min >= 0
        assert self.z_height >= self.z_min
        assert self.z_height <= self.z_max

    def limit_z_height(self, z_value, z_velocity):
        """限制z轴高度在安全范围内"""
        if z_value < self.z_min:
            z_value = self.z_min
            z_velocity = 0
        elif z_value > self.z_max:
            z_value = self.z_max
            z_velocity = 0
        return z_value, z_velocity

    def vins_callback(self, msg):
        """vins_fusion话题回调函数，获取当前位置和姿态"""
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y
        self.current_position.z = msg.pose.pose.position.z
        
        # 从四元数计算偏航角
        vins_q_w = msg.pose.pose.orientation.w
        vins_q_x = msg.pose.pose.orientation.x
        vins_q_y = msg.pose.pose.orientation.y
        vins_q_z = msg.pose.pose.orientation.z
        
        # 计算偏航角
        siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y)
        cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.position_received = True

    def calculate_target_state(self, t):
        """计算目标状态（位置、速度、yaw）"""
        # 无限往返模式处理
        if self.infinite_round_trip and self.flight_mode != "yaw_rotation":
            # 往返一次的总时间
            round_trip_time = 2 * self.total_time
            # 当前在哪个往返周期
            cycle = int(t / round_trip_time)
            # 在当前周期内的相对时间
            t_in_cycle = t % round_trip_time
            # 判断是正向还是反向
            if t_in_cycle < self.total_time:
                # 正向运动（从起点到终点）
                t_relative = t_in_cycle
                current_direction = self.direction
            else:
                # 反向运动（从终点回到起点）
                t_relative = t_in_cycle - self.total_time
                current_direction = -self.direction
        else:
            # 单次运动模式
            t_relative = t
            current_direction = self.direction
        
        # 先计算速度（只算一遍）
        if self.flight_mode == "yaw_rotation":
            # 旋转模式：位置速度为零
            velocity = 0.0
        else:
            if self.motion_mode == "constant_speed":
                # 匀速模式：始终以最大速度运动
                velocity = self.max_velocity
            else:
                # 变速模式：先加速后匀速
                if t_relative <= self.acceleration_time:
                    # 加速阶段：匀加速
                    velocity = (self.max_velocity / self.acceleration_time) * t_relative
                else:
                    # 匀速阶段
                    velocity = self.max_velocity
        
        # 用计算好的速度算距离
        if self.motion_mode == "constant_speed":
            # 匀速模式：距离 = 速度 × 时间
            distance = velocity * t_relative
        else:
            # 变速模式：先加速后匀速
            if t_relative <= self.acceleration_time:
                # 加速阶段：匀加速运动
                # 注意：这里不能用 velocity * t，因为velocity是瞬时速度
                # 需要用积分：距离 = ∫v(t)dt = ∫(a*t)dt = 0.5*a*t²
                distance = 0.5 * (self.max_velocity / self.acceleration_time) * t_relative * t_relative
            else:
                # 匀速阶段：加速距离 + 匀速距离
                accel_distance = 0.5 * self.max_velocity * self.acceleration_time
                const_distance = self.max_velocity * (t_relative - self.acceleration_time)
                distance = accel_distance + const_distance
        
        # 无限往返模式：反向运动时，需要从终点位置开始计算
        if self.infinite_round_trip and self.flight_mode != "yaw_rotation":
            if t_in_cycle >= self.total_time:
                # 反向运动：计算单程最大距离，然后从终点往回走
                if self.motion_mode == "constant_speed":
                    max_distance = self.max_velocity * self.total_time
                else:
                    # 变速模式：加速距离 + 匀速距离
                    accel_distance = 0.5 * self.max_velocity * self.acceleration_time
                    const_distance = self.max_velocity * (self.total_time - self.acceleration_time)
                    max_distance = accel_distance + const_distance
                # 从终点往回走的距离
                distance = max_distance - distance
        
        # 根据飞行模式计算位置和yaw（速度已计算好）
        if self.flight_mode == "horizontal":
            # 横向飞行：X轴运动，Y轴为0，Z轴为设定高度
            position = Point(self.start_x + distance * current_direction, self.start_y, self.z_height)
            velocity_vec = Point(velocity * current_direction, 0, 0)
            yaw = self.start_yaw
        elif self.flight_mode == "vertical":
            # 竖向飞行：Y轴运动，X和Z轴为0
            position = Point(self.start_x, self.start_y + distance * current_direction, self.z_height)
            velocity_vec = Point(0, velocity * current_direction, 0)
            yaw = self.start_yaw
        elif self.flight_mode == "z_offset":
            # Z轴偏移测试：Z轴运动，X和Y轴为0
            position = Point(self.start_x, self.start_y, self.z_height + distance * current_direction)
            velocity_vec = Point(0, 0, velocity * current_direction)  # Z轴offset模式：X轴速度
            yaw = self.start_yaw
        elif self.flight_mode == "yaw_rotation":
            # 原地旋转：保持位置不变，只改变yaw角
            position = Point(self.start_x, self.start_y, self.z_height)
            velocity_vec = Point(0, 0, 0)  # 位置速度为零
            # 计算yaw角（无限往返模式下持续旋转）
            if self.infinite_round_trip:
            yaw_radians = np.radians(self.yaw_rotation_angle)
            yaw_rate = yaw_radians / self.total_time
            yaw = self.start_yaw + yaw_rate * t
            else:
                yaw_radians = np.radians(self.yaw_rotation_angle)
                yaw_rate = yaw_radians / self.total_time
                yaw = self.start_yaw + yaw_rate * t_relative
        else:
            # 默认情况
            position = Point(self.start_x, self.start_y, self.z_height)
            velocity_vec = Point(0, 0, 0)
            yaw = self.start_yaw
        
        return position, velocity_vec, yaw
    
    def publish_position_command(self, position, velocity, yaw):
        """发布位置命令"""
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.trajectory_id = 1
        
        cmd.position = position
        cmd.velocity = velocity
        cmd.acceleration = Point(0, 0, 0)
        cmd.yaw = yaw
        cmd.yaw_dot = 0.0
        
        # 增加z轴限制，防止指令中的z超出安全范围
        if hasattr(cmd.position, "z"):
            cmd.position.z, cmd.velocity.z = self.limit_z_height(cmd.position.z, cmd.velocity.z)
        self.pos_cmd_pub.publish(cmd)
    
    def run_test(self):
        """运行测试"""
        print("开始速度偏移测试...")
        print(f"等待{self.start_delay}秒后开始...")
        rospy.sleep(self.start_delay)
        
        # 等待接收到vins_fusion位置信息
        print("等待接收vins_fusion位置信息...")
        while not self.position_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
            print("等待vins_fusion位置信息...", end='\r')
        
        if rospy.is_shutdown():
            return
            
        print(f"\n已接收到位置信息：({self.current_position.x:.3f}, {self.current_position.y:.3f}, {self.current_position.z:.3f}, {self.current_yaw:.3f})")
        
        # 第一阶段：飞到指定高度
        print(f"第一阶段：飞到指定高度 {self.initial_height}米...")
        self.start_x = self.current_position.x
        self.start_y = self.current_position.y
        self.start_yaw = self.current_yaw
        initial_vel = Point(0, 0, 0)
        
        # 保持初始位置直到到达指定高度
        height_reach_steps = int(self.height_hold_time * self.control_frequency)
        for _ in range(height_reach_steps):
            self.publish_position_command(Point(self.start_x, self.start_y, self.initial_height), initial_vel, self.start_yaw)
            rospy.sleep(1.0 / self.control_frequency)
                
        # 第二阶段：开始速度偏移测试
        if self.infinite_round_trip:
            print("第二阶段：开始无限往返测试（按Ctrl+C停止）...")
        else:
        print("第二阶段：开始速度偏移测试...")
        self.start_time = rospy.Time.now()
        rate = rospy.Rate(self.control_frequency)  # 使用配置的控制频率
        
        last_cycle = -1  # 用于跟踪往返周期
        
        while not rospy.is_shutdown():
            current_time = (rospy.Time.now() - self.start_time).to_sec()
            
            # 无限往返模式下不停止，单次模式下检查是否完成
            if not self.infinite_round_trip:
            if current_time >= self.total_time:
                print("测试完成！")
                self.test_completed = True
                break
            
            # 计算目标状态（位置、速度、yaw）
            target_pos, target_vel, target_yaw = self.calculate_target_state(current_time)
            
            # 发布位置命令
            self.publish_position_command(target_pos, target_vel, target_yaw)
            
            # 打印进度
            if self.verbose_output:
                if self.infinite_round_trip:
                    # 无限往返模式：显示往返周期信息
                    round_trip_time = 2 * self.total_time
                    cycle = int(current_time / round_trip_time)
                    t_in_cycle = current_time % round_trip_time
                    if cycle != last_cycle:
                        print(f"开始第 {cycle + 1} 次往返...")
                        last_cycle = cycle
                    
                    if int(current_time * 10) % 10 == 0:  # 每0.1秒打印一次
                        direction_str = "正向" if t_in_cycle < self.total_time else "反向"
                        print(f"往返周期: {cycle + 1}, {direction_str}, "
                              f"周期内时间: {t_in_cycle:.1f}/{round_trip_time:.1f}s, "
                              f"位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f}), "
                              f"速度: ({target_vel.x:.3f}, {target_vel.y:.3f}, {target_vel.z:.3f}), "
                              f"Yaw: {np.degrees(target_yaw):.1f}°")
                else:
                    # 单次模式：显示常规进度
                    if int(current_time * 10) % 10 == 0:  # 每0.1秒打印一次
                print(f"测试进度: {current_time:.1f}/{self.total_time:.1f}s, "
                      f"位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f}), "
                      f"速度: ({target_vel.x:.3f}, {target_vel.y:.3f}, {target_vel.z:.3f}), "
                      f"Yaw: {np.degrees(target_yaw):.1f}°")
            
            rate.sleep()
        
        # 测试完成后保持最后位置（仅单次模式）
        if self.test_completed and not self.infinite_round_trip:
            print(f"保持最终位置{self.final_position_hold_time}秒...")
            final_pos, _, final_yaw = self.calculate_target_state(self.total_time)
            final_vel = Point(0, 0, 0)  # 保持静止
            
            hold_steps = int(self.final_position_hold_time * self.control_frequency)
            for _ in range(hold_steps):
                self.publish_position_command(final_pos, final_vel, final_yaw)
                rospy.sleep(1.0 / self.control_frequency)
    
def main():
    """主函数"""
    # 创建测试实例
    test = VelocityOffsetTest()
    
    # 运行测试
    test.run_test()
    


if __name__ == '__main__':
    main()
