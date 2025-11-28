#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VINS to MAVROS Odometry Converter
将VINS的Odometry消息转换为MAVROS格式，以固定频率发布所有信息
"""

import rospy
import threading
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class VinsToMavrosOdom:
    def __init__(self):
        rospy.init_node('vins_to_mavros_odom', anonymous=True)
        
        # 读取参数
        self.vins_odom_topic = rospy.get_param('~vins_odom_topic', '/vins_fusion/imu_propagate')
        # self.vins_odom_topic = rospy.get_param('~vins_odom_topic', '/svo/pose_imu')
        self.vision_pose_topic = rospy.get_param('~vision_pose_topic', '/mavros/vision_pose/pose')
        self.publish_rate = rospy.get_param('~publish_rate', 30.0)  # Hz
        
        # 用于存储转换后的位置和姿态
        self.p_mav = np.array([0.0, 0.0, 0.0])
        self.q_mav = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.odom_lock = threading.Lock()
        
        # 创建订阅者和发布者
        self.vins_sub = rospy.Subscriber(self.vins_odom_topic, Odometry, self.vins_callback, queue_size=10)
        self.vision_pub = rospy.Publisher(self.vision_pose_topic, PoseStamped, queue_size=10)
        
        # 发布速率
        self.rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo("VINS to MAVROS Odometry converter initialized")
        rospy.loginfo("Subscribe: %s -> Publish: %s @ %.1fHz", self.vins_odom_topic, self.vision_pose_topic, self.publish_rate)
        
        # 启动发布循环
        self.publish_loop()
    
    def vins_callback(self, msg):
        """
        VINS消息回调函数
        进行坐标系转换：VINS (前x,左y,上z) -> MAVROS (前y,右x,上z)
        """
        if msg.header.frame_id == "world":
            with self.odom_lock:
                # 位置转换：NWU (北西上) -> ENU (东北上)
                # x(N) -> y, y(W) -> -x, z(U) -> z
                self.p_mav[0] = -msg.pose.pose.position.y  # W -> -E
                self.p_mav[1] = msg.pose.pose.position.x      # N -> N
                self.p_mav[2] = msg.pose.pose.position.z     # U -> U
                
                # 姿态转换：绕z轴旋转90度，从NWU转换到ENU
                q_vins = np.array([
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z
                ])
                
                # 创建绕z轴旋转90度的四元数
                yaw_angle = np.pi / 2.0  # 旋转90度
                q_rot = np.array([
                    np.cos(yaw_angle / 2.0),  # w
                    0.0,                       # x
                    0.0,                       # y
                    np.sin(yaw_angle / 2.0)   # z
                ])
                
                # 四元数乘法
                self.q_mav = self.quaternion_multiply(q_vins, q_rot)
    
    def quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        return np.array([w, x, y, z])
    
    def publish_loop(self):
        """固定频率发布循环"""
        while not rospy.is_shutdown():
            vision = PoseStamped()
            
            with self.odom_lock:
                vision.pose.position.x = self.p_mav[0]
                vision.pose.position.y = self.p_mav[1]
                vision.pose.position.z = self.p_mav[2]
                
                vision.pose.orientation.w = self.q_mav[0]
                vision.pose.orientation.x = self.q_mav[1]
                vision.pose.orientation.y = self.q_mav[2]
                vision.pose.orientation.z = self.q_mav[3]
            
            vision.header.stamp = rospy.Time.now()
            
            # 发布
            self.vision_pub.publish(vision)
            
            self.rate.sleep()


def main():
    try:
        converter = VinsToMavrosOdom()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

