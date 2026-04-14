#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集采集脚本：同步保存无人机 body 系原点坐标（GT）与深度相机 RGBD 图像。
同一时刻的 image 与 label 使用相同序号，保证一一对应。
"""

from __future__ import print_function

import os
import sys
import argparse
import json
from datetime import datetime
import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


def world_origin_in_body_frame(p_world, qw, qx, qy, qz):
    """
    由世界系下飞机位姿，求世界系原点在机体系下的坐标。
    p_world: 飞机（body 原点）在世界系下的位置 (3,)
    qw,qx,qy,qz: 机体在世界系下的四元数 (body to world)
    返回: 世界系原点在机体系下的坐标 (x, y, z)
    """
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])
    # R 为 body->world，世界原点在 body 系下 = R^T * (0 - p_world) = -R^T @ p_world
    p_world = np.array([p_world[0], p_world[1], p_world[2]], dtype=float)
    p_body = -R.T.dot(p_world)
    return float(p_body[0]), float(p_body[1]), float(p_body[2])


class RGBDGTCollector:
    def __init__(self, output_dir, odom_topic, color_topic, depth_topic,
                 save_depth_as_png=False, sync_slop=0.05, max_rate_hz=0, odom_msg_type="odometry"):
        self.output_dir = os.path.abspath(output_dir)
        self.odom_topic = odom_topic
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.save_depth_as_png = save_depth_as_png  # False=存 npy（推荐，完整量程无截断）；True=存 16 位 PNG
        self.sync_slop = sync_slop
        self.max_rate_hz = max_rate_hz  # 0 表示不限制

        self.bridge = CvBridge()
        self.frame_id = 0
        self.last_save_time = 0.0

        # 位姿话题消息类型：Odometry 或 PoseWithCovarianceStamped（如 /svo/pose_imu）
        if odom_msg_type == "pose_cov_stamped":
            self._odom_msg_type = PoseWithCovarianceStamped
        else:
            self._odom_msg_type = Odometry

        self.image_dir = os.path.join(self.output_dir, "images")
        self.depth_dir = os.path.join(self.output_dir, "depths")
        self.label_dir = os.path.join(self.output_dir, "labels")
        for d in [self.image_dir, self.depth_dir, self.label_dir]:
            os.makedirs(d, exist_ok=True)

        # 同步订阅（位姿用实际话题类型）
        self.odom_sub = message_filters.Subscriber(odom_topic, self._odom_msg_type, queue_size=10)
        self.color_sub = message_filters.Subscriber(color_topic, Image, queue_size=10)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=10)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.color_sub, self.depth_sub],
            queue_size=20,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, odom_msg, color_msg, depth_msg):
        now = rospy.Time.now().to_sec()
        if self.max_rate_hz > 0:
            if now - self.last_save_time < 1.0 / self.max_rate_hz:
                return
        self.last_save_time = now

        # 原始数据：世界系下的飞机位置与姿态
        px = odom_msg.pose.pose.position.x
        py = odom_msg.pose.pose.position.y
        pz = odom_msg.pose.pose.position.z
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        # 转换为：世界系原点在机体系下的坐标 (x, y, z)，作为 GT 保存
        x, y, z = world_origin_in_body_frame((px, py, pz), qw, qx, qy, qz)

        # 转 RGB 图像
        try:
            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, "RGB 转换失败: %s", e)
            return

        # 转深度图像（深度必须用 uint16 或 float32，不能用 uint8，否则量程/精度都不够）
        try:
            if depth_msg.encoding in ("16UC1", "16U", "z16", "Z16"):
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_np = np.array(depth, dtype=np.uint16)
            elif depth_msg.encoding in ("32FC1", "32F"):
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_np = np.array(depth, dtype=np.float32)
            else:
                depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                # 若上游是 8UC1 等 uint8，强制转为 uint16（按毫米解释，避免误存为 8 位）
                if depth_np.dtype == np.uint8:
                    rospy.logwarn_throttle(5.0, "深度图为 uint8，已转为 uint16（按毫米），建议使用 16UC1/32FC1 话题")
                    depth_np = np.array(depth_np, dtype=np.uint16)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "深度图转换失败: %s", e)
            return

        # 统一文件名序号
        name = "{:06d}".format(self.frame_id)

        # 保存 RGB（作为 image）
        rgb_path = os.path.join(self.image_dir, name + ".png")
        cv2.imwrite(rgb_path, rgb)

        # 保存深度：默认 npy 保留完整量程（uint16 0~65535 mm 或 float32 米），无截断；可选 16 位 PNG
        if self.save_depth_as_png:
            if depth_np.dtype == np.float32:
                scale = 1000.0  # m -> mm
                depth_mm = np.clip(depth_np * scale, 0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(self.depth_dir, name + ".png"), depth_mm)
            else:
                cv2.imwrite(os.path.join(self.depth_dir, name + ".png"), depth_np)
        else:
            np.save(os.path.join(self.depth_dir, name + ".npy"), depth_np)

        # 保存 label：仅 xyz 三坐标（json）
        label = {"x": float(x), "y": float(y), "z": float(z)}
        label_path = os.path.join(self.label_dir, name + ".json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, indent=2, ensure_ascii=False)

        print("[GT] %s  x=%.4f  y=%.4f  z=%.4f" % (name, x, y, z))

        self.frame_id += 1
        if self.frame_id % 50 == 0:
            rospy.loginfo("已保存 %d 帧 -> %s", self.frame_id, self.output_dir)


def main():
    # 默认目录带当前时间戳，每次运行不重复
    default_output = "./dataset/dataset_rgbd_gt_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="采集 RGBD + 无人机 body 原点 GT 数据集")
    parser.add_argument("--output_dir", "-o", type=str, default=default_output,
                        help="数据集根目录(含 images/depths/labels)；默认 dataset_rgbd_gt_年月日_时分秒，每次不重复")
    parser.add_argument("--odom_topic", type=str, default="/svo/pose_imu",
                        help="无人机位姿话题，body 原点即 pose.pose.position")
    parser.add_argument("--odom_type", type=str, choices=["odometry", "pose_cov_stamped"], default="pose_cov_stamped",
                        help="位姿话题消息类型: odometry=nav_msgs/Odometry, pose_cov_stamped=geometry_msgs/PoseWithCovarianceStamped (如 /svo/pose_imu 用后者)")
    parser.add_argument("--color_topic", type=str, default="/camera/color/image_raw",
                        help="RGB 图像话题")
    parser.add_argument("--depth_topic", type=str, default="/camera/depth/image_rect_raw",
                        help="深度图话题")
    parser.add_argument("--save_depth_png", action="store_true",
                        help="深度图存为 16 位 PNG；不指定则默认存 npy（推荐，完整量程无截断）")
    parser.add_argument("--sync_slop", type=float, default=0.05,
                        help="ApproximateTime 同步时间容差（秒）")
    parser.add_argument("--max_rate", type=float, default=2,
                        help="最大保存频率 Hz，0 表示不限制")
    args = parser.parse_args()

    rospy.init_node("collect_rgbd_gt_dataset", anonymous=True)
    c = RGBDGTCollector(
        output_dir=args.output_dir,
        odom_topic=args.odom_topic,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        save_depth_as_png=args.save_depth_png,
        sync_slop=args.sync_slop,
        max_rate_hz=args.max_rate,
        odom_msg_type=args.odom_type,
    )
    rospy.loginfo("采集已启动: GT=%s, RGB=%s, Depth=%s -> %s",
                  args.odom_topic, args.color_topic, args.depth_topic, args.output_dir)
    rospy.spin()


if __name__ == "__main__":
    main()
