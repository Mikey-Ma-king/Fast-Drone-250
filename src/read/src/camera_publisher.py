#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class CameraPublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('camera_publisher', anonymous=True)
        
        # 创建图像发布者
        self.image_pub = rospy.Publisher('/camera', Image, queue_size=1)
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 相机参数
        self.camera_id = rospy.get_param('~camera_id', 0)  # 默认使用相机0
        self.frame_width = rospy.get_param('~frame_width', 1280)  # 默认宽度
        self.frame_height = rospy.get_param('~frame_height', 720)  # 默认高度
        self.fps = rospy.get_param('~fps', 120)  # 默认帧率
        
        # 初始化相机 - 模仿read.cpp的方式
        self.cap = cv2.VideoCapture()
        
        # 尝试打开相机，从0到8逐个尝试
        camera_opened = False
        for i in range(9):
            if self.cap.open(i, cv2.CAP_V4L2):
                rospy.loginfo(f"成功打开相机 {i}")
                camera_opened = True
                break
            else:
                rospy.logwarn(f"无法打开相机 {i}")
        
        if not camera_opened:
            rospy.logerr("无法打开任何相机")
            return
            
        # 设置相机参数 - 模仿read.cpp的设置
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 获取实际相机参数
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        rospy.loginfo(f"相机初始化成功:")
        rospy.loginfo(f"  相机ID: {self.camera_id}")
        rospy.loginfo(f"  分辨率: {actual_width}x{actual_height}")
        rospy.loginfo(f"  帧率: {actual_fps}")
        rospy.loginfo(f"  发布话题: /camera")
        
        # 设置循环频率
        self.rate = rospy.Rate(self.fps)
        
    def publish_camera_feed(self):
        """发布相机图像流"""
        frame_count = 0
        start_time = time.time()
        
        while not rospy.is_shutdown():
            # 读取相机帧
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("无法读取相机帧")
                continue
                
            # 在图像上添加信息
            frame_with_info = frame.copy()
            cv2.putText(frame_with_info, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_with_info, f"Time: {rospy.Time.now().to_sec():.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 转换为ROS图像消息
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame_with_info, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "camera_frame"
                
                # 发布图像
                self.image_pub.publish(ros_image)
                
                frame_count += 1
                
                # 计算实际帧率
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    actual_fps = frame_count / elapsed_time
                    rospy.loginfo(f"实际发布帧率: {actual_fps:.2f} FPS")
                    
            except Exception as e:
                rospy.logerr(f"图像转换或发布失败: {e}")
                
            # 控制发布频率
            self.rate.sleep()
            
    def cleanup(self):
        """清理资源"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("相机发布器已关闭")

def main():
    try:
        camera_pub = CameraPublisher()
        camera_pub.publish_camera_feed()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'camera_pub' in locals():
            camera_pub.cleanup()

if __name__ == '__main__':
    main()
