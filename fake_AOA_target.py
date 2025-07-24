#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3
import math
import tf
import time
import random
import numpy as np
vins_p = np.array([0,0,0])
vins_v = np.array([0,0,0])
vins_yaw = 0
class UAVStateListener:
    def __init__(self):
        #rospy.init_node('uav_state_listener', anonymous=True)
        self.T1 = None
        self.R1 = None
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.pose_cb)
        self.current_position = None
        self.current_orientation = None
        self.current_rotation_matrix = None
        self.list = []
        self.yaw = 0
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb(self, msg):
        global vins_p
        global vins_yaw
        global vins_v
        # 从 Odometry 消息中获取位置和姿态
        self.list.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        if len(self.list)==3:
            self.T1 = np.array([np.mean(self.list,axis = 0)])
            quaternion = (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            vins_q_w = msg.pose.pose.orientation.w
            vins_q_x = msg.pose.pose.orientation.x
            vins_q_y = msg.pose.pose.orientation.y
            vins_q_z = msg.pose.pose.orientation.z
            siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y)
            cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
        # 设置触发条件的标志为True
            self.trigger_condition_met = True
        # 取消订阅
            self.list = []
            vins_p = np.array(self.T1[0])
            vins_yaw = self.yaw
            vins_v[0] = msg.twist.twist.linear.x
            vins_v[1] = msg.twist.twist.linear.y

def publish_circle_motion():
    global vins_p
    global vins_yaw
    global vins_v
    vins_listener = UAVStateListener()
    distance = 0.0
    rospy.init_node('target_circle_publisher')
    pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    target_pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    AOA_pub = rospy.Publisher('AOA_Tag_data', Odometry, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    radius = 4.0          # 圆的半径
    speed = 1.0           # 线速度 1 m/s
    omega = speed / radius  # 角速度 rad/s
    start_time = time.time()

    x = 0.0
    y = 0
    vx = 0.2
    vy = 0.1
    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        # print(t)

        # 圆周运动轨迹 (x = r*cos(θ), y = r*sin(θ)), 角度从 t * omega 增长
        theta = omega * t
        # x = radius * math.cos(theta) + 1.0  # x=1 为起始偏移
        # y = radius * math.sin(theta)
        # z = 1.0  # 保持恒定高度
        x += 1/30 * (vx) + random.gauss(0, 0.01)  # x=1 为起始偏移
        # print(x)
        y += 1/30 * (vy) + random.gauss(0, 0.01)
        z = 1.0  # 保持恒定高度
        vx += 0.1*random.gauss(0, 0.01)
        vy += 0.1*random.gauss(0, 0.01)
        vx = min(max(vx,-1),1)
        vy = min(max(vy,-1),1)

        # 速度方向
        # vx = -speed * math.sin(theta)
        # vy = speed * math.cos(theta)
        # vz = 0.0

        # 朝向的四元数（绕 z 轴）
        yaw = theta   # 方向与切线一致
        while abs(yaw) >math.pi:
            yaw -= 2*math.pi
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

        odom = Odometry()
        # odom.header.stamp = current_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "target_base"

        odom.pose.pose.position = Point(x, y, z)
        odom.pose.pose.orientation.w = math.atan2(vy, vx)
        odom.pose.pose.orientation.x  = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0

        odom.twist.twist.linear = Vector3(vx, vy, 0)
        odom.twist.twist.angular = Vector3(0, 0, omega)

        pub.publish(odom)
        target_pub.publish(odom)

        distance = math.sqrt((vins_p[0] - x)**2 + (vins_p[1] - y)**2 + 0*(vins_p[2] - z)**2)
        # print("distance",distance)

        AOA_msg = Odometry()
        odom.header.frame_id = "world"
        odom.child_frame_id = "AOA_distance"
        AOA_msg.pose.pose.position.x = distance
        AOA_msg.pose.pose.orientation.w = math.atan2(y-vins_p[1],x-vins_p[0]) - vins_yaw
        AOA_pub.publish(AOA_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
