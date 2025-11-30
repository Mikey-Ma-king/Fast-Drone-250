import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline

class DroneMPC:
    def __init__(self, N=10, dt=0.1):
        self.N = N
        self.dt = dt
        self.nx = 6  # [px,py,pz,vx,vy,vz]
        self.nu = 4  # [ax,ay,az,yaw_rate]

        # 权重矩阵
        self.Q = np.diag([10, 10, 20, 1, 1, 1])
        self.R = np.diag([0.1, 0.1, 0.1, 0.05])
        self.S = np.diag([0.5, 0.5, 0.5, 0.2])

        # 动力学模型
        self.A = np.eye(6)
        self.A[0:3, 3:6] = np.eye(3) * dt
        
        # B矩阵构建（修正后）
        B_acc = 0.5 * dt**2 * np.eye(3)
        B_vel = dt * np.eye(3)
        B_yaw = np.zeros((3, 1))
        self.B = np.hstack([np.vstack([B_acc, B_vel]), np.vstack([B_yaw, B_yaw])])
        
        # 约束
        self.v_max = np.array([1.2, 1.2, 0.8])
        self.a_max = np.array([1.2, 1.2, 1])
        self.yaw_rate_max = np.radians(30)

        self.w_theta = 0.1
    
    def solve(self, x0, x_ref_traj , target_pos=None, target_position=None, position_weight=0.0):
        x = cp.Variable((self.N+1, self.nx))
        u = cp.Variable((self.N, self.nu))
        cost = 0
        constraints = [x[0] == x0]

        if target_pos is None:
            target_x = x_ref_traj[0, 0]
            target_y = x_ref_traj[0, 1]
        else:
            target_x, target_y = target_pos[0], target_pos[1]
        
        for k in range(self.N):
            cost += cp.quad_form(x[k] - x_ref_traj[k], self.Q)
            cost += cp.quad_form(u[k], self.R)
            if k > 0:
                cost += cp.quad_form(u[k] - u[k-1], self.S)

            vx = x[k, 3]  
            vy = x[k, 4] 
            dx = target_x - x[k, 0]  
            dy = target_y - x[k, 1] 
            directional_error = vy * dx - vx * dy  # 叉积形式（速度方向与目标方向垂直分量的平方）
            cost += self.w_theta * cp.square(directional_error)
            
            # 位置约束：当前坐标与目标坐标的差的平方，乘以系数（只使用x和y）
            if target_position is not None and position_weight > 0:
                position_error_xy = x[k, 0:2] - target_position  # 只使用x和y
                cost += position_weight * cp.sum_squares(position_error_xy)  
            constraints += [
                x[k+1] == self.A @ x[k] + self.B @ u[k],
                cp.abs(x[k, 3:6]) <= self.v_max,
                cp.abs(u[k, :3]) <= self.a_max,
                cp.abs(u[k, 3]) <= self.yaw_rate_max
            ]
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False ,max_iter=5000, eps_abs=5e-6, eps_rel=5e-6)
        return x.value, u.value if prob.status == cp.OPTIMAL else (None, None)

