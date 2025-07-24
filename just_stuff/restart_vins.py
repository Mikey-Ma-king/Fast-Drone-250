#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool

def main():
    # 初始化ROS节点
    rospy.init_node('restart_vins_publisher', anonymous=True)

    # 创建Publisher，发布到/vins_restart话题，消息类型为std_msgs/Bool
    pub = rospy.Publisher('/vins_restart', Bool, queue_size=10)

    # 设置发布频率为10Hz
    rate = rospy.Rate(10)  # 10Hz

    rospy.loginfo("Starting to publish /vins_restart at 10Hz frequency...")

    # 循环发布消息
    while not rospy.is_shutdown():
        msg = Bool()
        msg.data = True
        pub.publish(msg)
        rate.sleep()  # 按照10Hz的频率休眠

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass