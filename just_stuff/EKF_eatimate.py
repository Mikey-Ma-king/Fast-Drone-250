import numpy as np
import math

###############################
# 全局变量 (假设外部实时更新)
###############################
target_p = np.array([2.0, -3.0])   # 目标的大致位置 (X, Y) (示例值)
target_v = np.array([1.0, 0.0])    # 目标的大致速度 (Vx, Vy) (示例值)

vins_p   = np.array([0.0, 0.0])    # 人(传感器)的当前位置 (X, Y)
distance = 0.0                    # 当前测得的人到目标的距离

###############################
# 扩展卡尔曼滤波器类
###############################
class EKFTracker:
    """
    追踪 2D 平面内目标的 (X, Y, Vx, Vy).
    测量模型: 只测到目标与传感器之间的距离 (range-only).
    """

    def __init__(self, init_state, init_cov, Q, R, dt=1.0):
        """
        :param init_state: shape=(4,) [X, Y, Vx, Vy] 的初始状态估计
        :param init_cov:   shape=(4,4) 初始协方差
        :param Q:          shape=(4,4) 过程噪声协方差
        :param R:          测量噪声标量 (或 1x1 矩阵)，表示距离测量噪声方差
        :param dt:         时间步
        """
        self.x = init_state.astype(float)  # 状态向量: [X, Y, Vx, Vy]
        self.P = init_cov.astype(float)    # 状态协方差 4x4
        self.Q = Q.astype(float)           # 过程噪声协方差
        self.R = float(R)                  # 距离测量噪声方差(标量)
        self.dt = dt

        # 匀速运动模型的状态转移矩阵 F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def predict(self):
        """
        EKF 预测步骤: x(t|t-1), P(t|t-1).
        """
        # 1) 状态预测
        self.x = self.F @ self.x
        # 2) 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, sensor_pos, measured_range):
        """
        EKF 更新步骤: 只基于测量距离(标量).
        :param sensor_pos: shape=(2,) 传感器(人)位置 [x_s, y_s]
        :param measured_range: float, 测得距离
        """
        # 读取当前预测值
        X_pred, Y_pred, Vx_pred, Vy_pred = self.x

        # 1) 计算测量预测值 z_hat = h(x)
        #    h(x) = sqrt((X - x_s)^2 + (Y - y_s)^2)
        dx = X_pred - sensor_pos[0]
        dy = Y_pred - sensor_pos[1]
        dist_pred = math.sqrt(dx*dx + dy*dy)

        # 如果距离非常小，避免除以零
        if dist_pred < 1e-8:
            return

        # 2) 计算测量雅可比 H (1 x 4)
        #    H = [ (X-x_s)/dist, (Y-y_s)/dist, 0, 0 ]
        H = np.array([[dx/dist_pred, dy/dist_pred, 0.0, 0.0]])

        # 3) 计算卡尔曼增益 K
        #    S = H * P * H^T + R  => 标量
        S = H @ self.P @ H.T + self.R
        K = (self.P @ H.T) / S  # shape=(4,1)

        # 4) 状态更新
        #    y = z - z_hat
        y = measured_range - dist_pred
        self.x = self.x + (K.reshape(-1) * y)

        # 5) 协方差更新
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

###############################
# 全局的 EKF 对象
###############################
ekf = None

def ekf_init():
    """
    初始化 EKF 对象。
    假设:
    - 目标的大致位置在 target_p
    - 目标的大致速度在 target_v
    - 它们都是全局变量.
    """
    global ekf, target_p, target_v

    # 1) 构造初始状态 (X, Y, Vx, Vy)：
    #    把 "目标大致位置" 和 "目标大致速度" 放进去
    init_state = np.array([
        target_p[0],
        target_p[1],
        target_v[0],
        target_v[1]
    ], dtype=float)

    # 2) 初始协方差，根据对先验信息的信任程度来设置
    #    例如, 对位置给大一些(方差=25 => 标准差~5m), 对速度给稍小一些(方差=9 => 标准差=3m/s).
    init_cov = np.diag([9.0, 9.0, 0.25, 0.25])

    # 3) 过程噪声Q, 表示目标的动态不确定性 (可以根据经验或场景调节)
    Q = np.diag([0.1, 0.1, 1.0, 1.0])

    # 4) 测量噪声R, 假设测距标准差约 0.2m => 方差=0.04
    R = 0.012

    # 5) 创建EKF对象, dt=1s (可根据实际刷新频率)
    ekf = EKFTracker(init_state, init_cov, Q, R, dt=1.0)

def ekf_step():
    """
    每当有新的全局测距数据(distance)和传感器(人)位置(vins_p)时,
    调用这个函数执行一次EKF的 predict+update.
    执行结束后, 就可从 ekf.x 中拿到目标最新的 (X, Y, Vx, Vy).
    """
    global ekf, vins_p, distance

    if ekf is None:
        print("EKF not initialized. Call ekf_init() first.")
        return

    # 1) EKF 预测
    ekf.predict()

    # 2) EKF 更新
    #    measured_range = distance
    #    sensor_pos = vins_p
    ekf.update(vins_p, distance)

    # （可选）在此打印或者记录一下最新估计
    # current_est = ekf.x
    # print("Estimated Target State:", current_est)

###############################
# DEMO: 如何使用
###############################
if __name__ == "__main__":
    # 1) 在程序启动时, 初始化EKF
    ekf_init()

    # 2) 假设外部系统会不断更新 vins_p / distance / target_p / target_v
    #    这里仅做一个演示，循环调用 ekf_step()
    for step in range(10):
        # 这里随便模拟一下 "人的位置" 与 "测得距离"
        # 实际使用时, 这两者在别的地方更新
        vins_p[:]   = np.array([step*0.5, step*0.2])  # 人的位移 (演示)
        distance    = 10.0 - step*0.3                # 人-目标距离 (演示)

        # 执行一次 EKF predict + update
        ekf_step()

        # 打印当前滤波器估计
        print(f"Step {step}: EKF Estimated State = {ekf.x}")
