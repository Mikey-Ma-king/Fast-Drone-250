#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time

# 初始化相机
cap = cv2.VideoCapture()

# 尝试打开相机
camera_opened = False
for i in range(9):
    if cap.open(i, cv2.CAP_V4L2):
        print(f"成功打开相机 {i}")
        camera_opened = True
        break

if not camera_opened:
    print("无法打开任何相机")
    exit()

# 设置相机参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为30FPS

# 显示相机画面
frame_count = 0
start_time = time.time()
actual_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # 计算实际帧率
    frame_count += 1
    if frame_count % 30 == 0:  # 每30帧显示一次帧率
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        print(f"实际帧率: {actual_fps:.2f} FPS")
    
    # 在画面上显示帧率
    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 转为灰度图像并增强对比度
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=20)  # 轻微提高对比度和亮度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Camera', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
