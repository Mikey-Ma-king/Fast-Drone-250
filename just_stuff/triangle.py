import numpy as np
from scipy.optimize import root_scalar
import math

def triangulate(d_A, d_B, d_AB, dog_yaw):
    """
    计算 A, B 的相对基站的坐标
    :param d_A: 基站到 A 的距离
    :param d_B: 基站到 B 的距离
    :param d_AB: A 和 B 之间的固定距离
    :param theta: 向量 AB 相对于 x 轴的角度 (弧度制)
    :return: A 和 B 的坐标
    """
    def f(theta,d_A,d_B,d_AB):
        return (d_B**2 - (d_A * math.cos(theta))**2)**0.5 - d_A * math.sin(theta) - d_AB

    best_theta = 0
    if (abs(d_B - d_A) > d_AB):
        best_theta = 0
        print("warning: d_B - d_A too long.")
    elif (abs(d_B**2 - d_A**2) < d_AB):
        best_theta = 3.14159/2
        print("warning: too long.")
    else:
        # 定义搜索范围（0 到 90 度）
        bracket = (0, math.pi / 2)
        # 使用 root_scalar 求解
        result = root_scalar(f, args=(d_A, d_B, d_AB), bracket=bracket, method='brentq')
        best_theta = result.root
    x_A = -d_A*math.sin(best_theta) - 0.5*d_AB
    y_A = -d_A*math.cos(best_theta)
    print(best_theta)

    return (x_A, y_A), (x_A, -y_A)

# 测试
d_A = 3.0   # 基站到 A 的距离
d_B = 3.2   # 基站到 B 的距离
d_AB = 0.4  # A 和 B 之间的距离
theta = np.radians(30)  # AB 向量相对 x 轴的角度 (30°)

A_pos, B_pos = triangulate(d_A, d_B, d_AB, theta)
print(f"A 的坐标: {A_pos}")
print(f"B 的坐标: {B_pos}")
