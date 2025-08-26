import cv2
import numpy as np
import glob
import os

# 设置棋盘格尺寸
chessboard_size = (11, 8)
square_size = 0.045
  # 单位：米

# 准备棋盘格点的3D坐标
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# 存储所有图像的3D点和2D点
objpoints = []
imgpoints = []

# 获取棋盘格图像
image_dir = 'captured_images_5'
images = glob.glob(os.path.join(image_dir, 'frame_*.jpg'))

if not images:
    print("未找到任何图像，请确保path_to_images目录下有标定图像。")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像文件: {fname}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(f"查找棋盘格角点结果: {ret} - 文件名: {fname}")

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"未找到棋盘格角点: {fname}")

cv2.destroyAllWindows()

if not objpoints or not imgpoints:
    print("没有找到任何有效的棋盘格角点，请检查标定图像。")
    exit()

# 标定摄像头
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印结果
print("Camera matrix : \n")
print(mtx)
print("Distortion coefficients : \n")
print(dist)

# 保存内参和畸变系数
np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
