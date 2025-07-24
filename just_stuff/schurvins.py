#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

class IMUPoseRepublisher:
    def __init__(self):
        # 订阅 /svo/pose_imu (geometry_msgs/PoseWithCovarianceStamped)，队列大小设为10
        self.sub_imu_pose = rospy.Subscriber(
            "/svo/pose_imu",
            PoseWithCovarianceStamped,
            self.imu_pose_callback,
            queue_size=1
        )

        # 发布 /svo/pose_odom (nav_msgs/Odometry)
        self.pub_odom = rospy.Publisher(
            "/vins_fusion/imu_propagate",
            Odometry,
            queue_size=1
        )

        # 用于存储最近一次收到的 IMU Pose 消息
        self.latest_imu_pose = None

    def imu_pose_callback(self, msg):
        """
        回调函数：当订阅到新的 PoseWithCovarianceStamped 消息时，保存到类成员变量。
        """
        self.latest_imu_pose = msg

    def publish_loop(self):
        """
        以 200Hz 频率循环发布 Odometry 消息
        """
        rate = rospy.Rate(200)  # 200 Hz
        while not rospy.is_shutdown():
            if self.latest_imu_pose is not None:
                # 1. 构造 Odometry 消息
                odom_msg = Odometry()

                # 复制头部信息
                odom_msg.header = self.latest_imu_pose.header
                # 你也可以自行修改 frame_id, child_frame_id
                # 例如:
                odom_msg.header.frame_id = "world"     # 世界坐标系
                odom_msg.child_frame_id = "imu_link"   # IMU 坐标系

                # 2. 填充 pose 信息（包括 covariance）
                odom_msg.pose = self.latest_imu_pose.pose  
                # nav_msgs/Odometry 中的 pose 与 geometry_msgs/PoseWithCovarianceStamped 中的 pose 字段对应

                # 3. 如果需要处理 twist，请在这里添加
                # 例如:
                # odom_msg.twist.twist.linear.x = ...
                # odom_msg.twist.twist.angular.z = ...
                # odom_msg.twist.covariance = ...

                # 4. 发布
                self.pub_odom.publish(odom_msg)

            # 按照设定的 200Hz 休眠
            rate.sleep()


def main():
    # 初始化节点
    rospy.init_node("imu_pose_republisher", anonymous=True)

    # 创建并运行节点
    republisher = IMUPoseRepublisher()
    republisher.publish_loop()


if __name__ == "__main__":
    main()
