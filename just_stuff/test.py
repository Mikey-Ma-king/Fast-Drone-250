import cv2
dabing = 0
# 打开视频文件
cap = cv2.VideoCapture('output_video_14:31:27.mp4')

# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 显示视频帧
    cv2.imshow('Video', frame)
    if dabing < -20:
        dabing += 1 
        continue
    
    # 等待1毫秒，按下'q'键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕捉器并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
# import math

# def calculate_fov(width, height, fx, fy):
#     """
#     根据摄像机内参计算水平和垂直视场角。

#     参数：
#     width -- 传感器的宽度（像素）
#     height -- 传感器的高度（像素）
#     fx -- 焦距（在水平方向上，单位通常是像素）
#     fy -- 焦距（在垂直方向上，单位通常是像素）

#     返回：
#     hfov -- 水平视场角（度）
#     vfov -- 垂直视场角（度）
#     """
#     hfov = 2 * math.atan2(width / 2.0, fx) * 180.0 / math.pi
#     vfov = 2 * math.atan2(height / 2.0, fy) * 180.0 / math.pi
#     return hfov, vfov

# # 示例使用
# width = 1920
# height = 1080
# fx = 1391.6163330078125
# fy = 1390.30322265625  # 假设的焦距值，需根据实际相机参数调整

# hfov, vfov = calculate_fov(width, height, fx, fy)
# print(f"Horizontal Field of View: {hfov} degrees")
# print(f"Vertical Field of View: {vfov} degrees")
