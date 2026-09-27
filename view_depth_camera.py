#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控深度相机 RGB + 深度图。

用法（需已 source devel/setup.bash，且相机驱动或仿真已在跑）:
  python3 view_depth_camera.py              # 按 agent/config.py 的 SIMULATE 选话题
  python3 view_depth_camera.py --real       # 真机 RealSense
  python3 view_depth_camera.py --sim        # Gazebo iris_realsense_camera
  python3 view_depth_camera.py --depth-only
  python3 view_depth_camera.py --list       # 列出当前 /camera 与 /iris 图像话题

快捷键: q / ESC 退出
"""

from __future__ import print_function

import argparse
import sys
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


REAL_RGB = "/camera/color/image_raw"
REAL_DEPTH = "/camera/depth/image_rect_raw"
SIM_RGB = "/iris/realsense/depth_camera/color/image_raw"
SIM_DEPTH = "/iris/realsense/depth_camera/depth/image_raw"


def _topics_from_config():
    try:
        from agent.config import DEPTH_IMAGE_TOPIC, RGB_IMAGE_TOPIC, SIMULATE

        return RGB_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, SIMULATE
    except Exception:
        return REAL_RGB, REAL_DEPTH, False


def depth_to_meters(depth_np, encoding):
    encoding = (encoding or "").lower()
    if depth_np.dtype == np.uint16 or encoding in ("16uc1", "16u", "mono16", "z16"):
        return depth_np.astype(np.float32) / 1000.0
    return depth_np.astype(np.float32)


def colorize_depth(depth_m, max_depth_m):
    vis = np.clip(depth_m, 0.0, max_depth_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    vis = (vis / max(max_depth_m, 1e-6) * 255.0).astype(np.uint8)
    vis[~valid] = 0
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def overlay_stats(bgr, lines):
    out = bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 22
    return out


def list_image_topics():
    names = rospy.get_published_topics()
    interesting = []
    for name, typ in names:
        if typ != "sensor_msgs/Image":
            continue
        if any(k in name for k in ("camera", "realsense", "depth", "color", "infra")):
            interesting.append(name)
    interesting.sort()
    if not interesting:
        print("当前没有图像话题。请先启动相机：")
        print("  真机: roslaunch realsense2_camera rs_camera.launch")
        print("  仿真: sh sim_fly.sh  （或确认 /iris/realsense/depth_camera/... 已发布）")
        return
    print("图像话题:")
    for n in interesting:
        print(" ", n)


class DepthCamViewer:
    def __init__(self, rgb_topic, depth_topic, depth_only, max_depth_m):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.depth_only = depth_only
        self.max_depth_m = max_depth_m
        self.bridge = CvBridge()
        self.rgb = None
        self.depth_m = None
        self.depth_encoding = ""
        self.rgb_stamp = 0.0
        self.depth_stamp = 0.0
        self.rgb_count = 0
        self.depth_count = 0
        self.t0 = time.time()

        if not depth_only:
            rospy.Subscriber(rgb_topic, Image, self._on_rgb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(depth_topic, Image, self._on_depth, queue_size=1, buff_size=2 ** 24)

    def _on_rgb(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.rgb_stamp = msg.header.stamp.to_sec()
            self.rgb_count += 1
        except Exception as e:
            rospy.logwarn_throttle(2.0, "RGB 转换失败: %s", e)

    def _on_depth(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_m = depth_to_meters(np.asarray(raw), msg.encoding)
            self.depth_encoding = msg.encoding
            self.depth_stamp = msg.header.stamp.to_sec()
            self.depth_count += 1
        except Exception as e:
            rospy.logwarn_throttle(2.0, "深度转换失败: %s", e)

    def _hz(self, count):
        dt = max(time.time() - self.t0, 1e-3)
        return count / dt

    def make_frame(self):
        depth_vis = None
        stats = []
        if self.depth_m is not None:
            depth_vis = colorize_depth(self.depth_m, self.max_depth_m)
            valid = self.depth_m[np.isfinite(self.depth_m) & (self.depth_m > 0.01)]
            h, w = self.depth_m.shape[:2]
            cy, cx = h // 2, w // 2
            center = float(self.depth_m[cy, cx])
            vmin = float(valid.min()) if valid.size else 0.0
            vmax = float(valid.max()) if valid.size else 0.0
            stats.append(
                "depth %s  %.1f Hz  %dx%d  %s"
                % (self.depth_topic, self._hz(self.depth_count), w, h, self.depth_encoding)
            )
            stats.append(
                "center=%.2fm  min=%.2fm  max=%.2fm  (jet 0-%.1fm)"
                % (center, vmin, vmax, self.max_depth_m)
            )
        else:
            stats.append("等待深度: %s" % self.depth_topic)

        if self.depth_only:
            if depth_vis is None:
                return overlay_stats(np.zeros((480, 640, 3), np.uint8), stats)
            return overlay_stats(depth_vis, stats)

        rgb = self.rgb
        if rgb is None:
            stats.insert(0, "等待 RGB: %s" % self.rgb_topic)
            canvas = depth_vis if depth_vis is not None else np.zeros((480, 640, 3), np.uint8)
            return overlay_stats(canvas, stats)

        rh, rw = rgb.shape[:2]
        stats.insert(
            0, "rgb %s  %.1f Hz  %dx%d" % (self.rgb_topic, self._hz(self.rgb_count), rw, rh)
        )
        if depth_vis is None:
            return overlay_stats(rgb, stats)

        dh, dw = depth_vis.shape[:2]
        if (dh, dw) != (rh, rw):
            depth_vis = cv2.resize(depth_vis, (rw, rh), interpolation=cv2.INTER_NEAREST)
        panel = np.hstack([rgb, depth_vis])
        return overlay_stats(panel, stats)


def main():
    cfg_rgb, cfg_depth, cfg_sim = _topics_from_config()
    parser = argparse.ArgumentParser(description="监控深度相机 RGB/深度图")
    parser.add_argument("--real", action="store_true", help="真机话题")
    parser.add_argument("--sim", action="store_true", help="仿真话题")
    parser.add_argument("--rgb", default="", help="覆盖 RGB 话题")
    parser.add_argument("--depth", default="", help="覆盖深度话题")
    parser.add_argument("--depth-only", action="store_true", help="只显示深度")
    parser.add_argument("--max-depth", type=float, default=8.0, help="伪彩色最大深度(米)")
    parser.add_argument("--list", action="store_true", help="列出图像话题后退出")
    args = parser.parse_args()

    rospy.init_node("view_depth_camera", anonymous=True)

    if args.list:
        list_image_topics()
        return

    if args.real:
        rgb_topic, depth_topic = REAL_RGB, REAL_DEPTH
    elif args.sim:
        rgb_topic, depth_topic = SIM_RGB, SIM_DEPTH
    else:
        rgb_topic, depth_topic = cfg_rgb, cfg_depth
        print("使用 agent/config.py：SIMULATE=%s" % cfg_sim)

    if args.rgb:
        rgb_topic = args.rgb
    if args.depth:
        depth_topic = args.depth

    print("RGB :", rgb_topic)
    print("Depth:", depth_topic)
    print("退出: q / ESC")
    print("若无画面，先确认话题在发:  rostopic hz", depth_topic)

    viewer = DepthCamViewer(rgb_topic, depth_topic, args.depth_only, args.max_depth)
    win = "depth_camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        frame = viewer.make_frame()
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("失败:", e, file=sys.stderr)
        sys.exit(1)
