import cv2
import time
import numpy as np

time.sleep(4)
# 打开摄像头，0 通常是默认的摄像头
cap = cv2.VideoCapture(6,cv2.CAP_V4L2)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)
#cap.set(cv2.CAP_PROP_FPS, 180)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
if cap.isOpened:
    print("可以打开")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# cap.set(cv2.CAP_PROP_FPS, 180)



save_path = './captured_images_4/'
prefix = 'image_'

# 每隔0.8秒拍摄一张图片
interval = 0.8  # 0.8秒
num_photos = 80
photo_count = 0

# 获取当前时间
start_time = time.time()

# 检查摄像头是否成功打开

# 实时显示摄像头画面并判断是否保存图像
while photo_count < num_photos:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # # 显示实时画面
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # 使用CLAHE（自适应直方图均衡化）增强对比度
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(gray)

    # # 使用高斯模糊降低噪声
    # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # # 使用锐化滤波器
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # sharpened = cv2.filter2D(blurred, -1, kernel)

    cv2.imshow('Camera', frame)

    # 检查是否到达保存图片的时间间隔
    elapsed_time = time.time() - start_time
    if elapsed_time >= interval:
        # 保存图像
        filename = f"{save_path}{prefix}{photo_count + 1}.jpg"
        cv2.imwrite(filename, frame)
        print(f"已保存 {filename}")

        # 重置计时器
        start_time = time.time()
        photo_count += 1

    # 按下 'q' 键可以退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
