import numpy as np
from scipy.optimize import minimize

# 生成模拟数据（假设真实偏移量为 true_delta_r）
np.random.seed(0)
N = 10
true_delta_r = np.array([1.5, -0.8])  # 真实的待求偏移量

# 生成机器狗位置（假设在2D平面运动）
dog_pos = np.random.rand(N, 2) * 10    # 随机生成10个位置
drone_pos = np.random.rand(N, 2) * 10    # 随机生成10个位置
drone_pos_true = drone_pos + true_delta_r  # 无人机的真实位置（无噪声）

distances = np.linalg.norm(drone_pos_true - dog_pos, axis=1)  # 带噪声的距离观测

# 定义目标函数
def objective(delta_r):
    """
    目标函数：最小化绝对误差和
    delta_r : 待优化的偏移量 [delta_x, delta_y]
    """
    delta_r = np.array(-delta_r)
    errors = []
    for i in range(N):
        # 计算修正后的理论距离
        corrected_distance = np.linalg.norm(drone_pos[i] - dog_pos[i] - delta_r)
        # 计算与实测距离的绝对误差
        errors.append(abs(corrected_distance - distances[i]))
    return np.sum(errors)

# 优化求解
initial_guess = [1.5, -0.8]  # 初始猜测值
result = minimize(objective, initial_guess, method='Nelder-Mead')

# 解析结果
optimal_delta_r = result.x
print(f"真实偏移量: {true_delta_r}")
print(f"优化结果: {optimal_delta_r}")
print(f"总绝对误差: {result.fun:.4f}")