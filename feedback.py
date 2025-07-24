#!/usr/bin/env python
import numpy as np
import rospy
import time
from nav_msgs.msg import Odometry

real_position = np.array([0,0,2])
t = 0
def position_callback(msg):
    global t
    # # 将接收到的位置信息更新到参数服务器
    # rospy.set_param('/drone0/planning/initial_px', float(msg.pose.pose.position.x))
    # rospy.set_param('/drone0/planning/initial_py', float(msg.pose.pose.position.y))
    # rospy.set_param('/drone0/planning/initial_pz', float(msg.pose.pose.position.z))
    if (t > 60):
        position = [msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]
        print("vins:",position)
        t = 0
    t += 1

def listener():
    global real_position
    # 初始化节点
    rospy.init_node('position_server_updater', anonymous=True)

    rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, position_callback, queue_size=5)

    # 保持程序运行直到节点被关闭
    rospy.spin()

if __name__ == '__main__':
    listener()
