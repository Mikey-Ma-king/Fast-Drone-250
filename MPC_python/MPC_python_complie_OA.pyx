# cython: language_level=3

import numpy as np
cimport numpy as np
import math
import osqp
from scipy import sparse


cdef class DroneMPC:
    cdef public int N, nx, nu
    cdef public double dt

    cdef public object Q, R, S
    cdef public object A, B

    cdef public object v_max, a_max, yaw_rate_max

    cdef public object P, q_template, A_cons, l_template, u_template
    cdef public object solver

    # 障碍 3D 点软避障（enable_obstacle=True）
    cdef public int obs_grid_n
    cdef public double obs_depth_max_m   # 仅 MPC.py 建点距离上限
    cdef public double obs_loss_weight

    def __init__(self, int N=10, double dt=0.1):
        self.N = N
        self.dt = dt
        self.nx = 6
        self.nu = 4

        self.Q = np.diag([10, 10, 20, 1, 1, 1])
        self.R = np.diag([0.1, 0.1, 0.1, 0.05])
        self.S = np.diag([0.5, 0.5, 0.5, 0.2])

        self.A = np.eye(6)
        self.A[0:3, 3:6] = np.eye(3) * dt

        cdef np.ndarray[np.float64_t, ndim=2] B_acc = 0.5 * dt**2 * np.eye(3)
        cdef np.ndarray[np.float64_t, ndim=2] B_vel = dt * np.eye(3)
        cdef np.ndarray[np.float64_t, ndim=2] B_yaw = np.zeros((3, 1))
        self.B = np.hstack([np.vstack([B_acc, B_vel]), np.vstack([B_yaw, B_yaw])])

        self.v_max = np.array([1.2, 1.2, 0.8])
        self.a_max = np.array([1.2, 1.2, 1.0])
        self.yaw_rate_max = np.radians(30)

        self.obs_grid_n = 8
        self.obs_depth_max_m = 2.0
        self.obs_loss_weight = 15.0      # slack 惩罚权重，实际 w = weight/sqrt(n_pts)

        self.P = None
        self.A_cons = None
        self.solver = None

    def _build_obstacle_slack_qp(
        self,
        object P_base,
        np.ndarray q_base,
        np.ndarray l_base,
        np.ndarray u_base,
        object A_base,
        np.ndarray x0,
        np.ndarray points,
        int N,
        int nx,
        int nu,
    ):
        """3D 障碍点 → 半空间 n·p <= ub + s，slack 惩罚 (w/sqrt(n_pts))·s²；n 指向障碍。"""
        cdef int m, k, si, row, n_valid, n_slack, n_var_base, n_var, n_cons_base, n_cons
        cdef int base, slack_var, n_cons_per_pt
        cdef double ox, oy, oz, d0, nx_d, ny_d, nz_d, ub, w_eff
        cdef np.ndarray[np.float64_t, ndim=1] nx_arr, ny_arr, nz_arr, ub_arr

        n_pts = points.shape[0]
        if n_pts == 0:
            return P_base, q_base, A_base, l_base, u_base

        w_eff = self.obs_loss_weight / np.sqrt(float(n_pts))
        n_cons_per_pt = 2 * N  # n·p - s <= ub 以及 s >= 0

        nx_arr = np.empty(n_pts, dtype=np.float64)
        ny_arr = np.empty(n_pts, dtype=np.float64)
        nz_arr = np.empty(n_pts, dtype=np.float64)
        ub_arr = np.empty(n_pts, dtype=np.float64)
        n_valid = 0
        for m in range(n_pts):
            ox = float(points[m, 0])
            oy = float(points[m, 1])
            oz = float(points[m, 2])
            d0 = math.hypot(ox - x0[0], math.hypot(oy - x0[1], oz - x0[2]))
            if d0 < 1e-6:
                continue
            nx_d = (ox - x0[0]) / d0
            ny_d = (oy - x0[1]) / d0
            nz_d = (oz - x0[2]) / d0
            nx_arr[n_valid] = nx_d
            ny_arr[n_valid] = ny_d
            nz_arr[n_valid] = nz_d
            ub_arr[n_valid] = nx_d * ox + ny_d * oy + nz_d * oz
            n_valid += 1

        if n_valid == 0:
            return P_base, q_base, A_base, l_base, u_base

        nx_arr = nx_arr[:n_valid]
        ny_arr = ny_arr[:n_valid]
        nz_arr = nz_arr[:n_valid]
        ub_arr = ub_arr[:n_valid]

        n_slack = n_valid * N
        n_var_base = (N + 1) * nx + N * nu
        n_var = n_var_base + n_slack
        n_cons_base = A_base.shape[0]
        n_cons = n_cons_base + n_valid * n_cons_per_pt

        P_slack = sparse.diags(
            np.full(n_slack, 2.0 * w_eff, dtype=np.float64),
            format="csc",
        )
        P_full = sparse.block_diag([P_base, P_slack], format="csc")

        q_full = np.zeros(n_var, dtype=np.float64)
        q_full[:n_var_base] = q_base

        A_pad = sparse.csc_matrix((n_cons_base, n_slack))
        A_obs = sparse.lil_matrix((n_valid * n_cons_per_pt, n_var))
        l_obs = np.empty(n_valid * n_cons_per_pt, dtype=np.float64)
        u_obs = np.empty(n_valid * n_cons_per_pt, dtype=np.float64)
        row = 0
        for m in range(n_valid):
            nx_d = float(nx_arr[m])
            ny_d = float(ny_arr[m])
            nz_d = float(nz_arr[m])
            ub = float(ub_arr[m])
            for k in range(1, N + 1):
                base = k * nx
                si = m * N + (k - 1)
                slack_var = n_var_base + si
                A_obs[row, base + 0] = nx_d
                A_obs[row, base + 1] = ny_d
                A_obs[row, base + 2] = nz_d
                A_obs[row, slack_var] = -1.0
                l_obs[row] = -np.inf
                u_obs[row] = ub
                row += 1
                A_obs[row, slack_var] = 1.0
                l_obs[row] = 0.0
                u_obs[row] = np.inf
                row += 1

        A_full = sparse.vstack(
            [sparse.hstack([A_base, A_pad]), A_obs.tocsc()],
            format="csc",
        )
        l_full = np.concatenate([l_base, l_obs])
        u_full = np.concatenate([u_base, u_obs])
        return P_full, q_full, A_full, l_full, u_full

    def solve(self, np.ndarray[np.float64_t, ndim=1] x0, np.ndarray[np.float64_t, ndim=2] x_ref_traj,
              target_pos=None, target_position=None, double position_weight=0.0,
              object obstacle_points=None, bint enable_obstacle=False):
        cdef int nx = self.nx
        cdef int nu = self.nu
        cdef int N = self.N
        cdef int n_var = (N+1) * nx + N * nu
        cdef np.ndarray[np.float64_t, ndim=1] target_pos_xy

        cdef int n_init_cons = nx
        cdef int n_dyn_cons = N * nx
        cdef int n_vel_cons = N * 3
        cdef int n_acc_cons = N * 3
        cdef int n_yaw_cons = N
        cdef int n_cons = n_init_cons + n_dyn_cons + n_vel_cons + n_acc_cons + n_yaw_cons

        cdef object P_lil, P_cur, A_cur
        cdef int idx
        cdef np.ndarray points = None
        cdef bint need_p_copy, use_obstacle_slack
        cdef np.ndarray l, u

        if self.P is None:
            Q_blocks = [self.Q for _ in range(N)] + [self.Q]
            R_blocks = [self.R for _ in range(N)]
            self.P = sparse.block_diag(Q_blocks + R_blocks).tocsc()

            rows = []

            row_init = np.zeros((nx, n_var))
            row_init[:, 0:nx] = np.eye(nx)
            rows.append(row_init)

            for k in range(N):
                row = np.zeros((nx, n_var))
                row[:, k*nx:(k+1)*nx] = -self.A
                row[:, (k+1)*nx:(k+2)*nx] = np.eye(nx)
                row[:, (N+1)*nx+k*nu:(N+1)*nx+(k+1)*nu] = -self.B
                rows.append(row)

            for k in range(N):
                for i in range(3):
                    row = np.zeros((1, n_var))
                    row[0, k*nx + 3 + i] = 1
                    rows.append(row)

            for k in range(N):
                for i in range(3):
                    row = np.zeros((1, n_var))
                    row[0, (N+1)*nx + k*nu + i] = 1
                    rows.append(row)

            for k in range(N):
                row = np.zeros((1, n_var))
                row[0, (N+1)*nx + k*nu + 3] = 1
                rows.append(row)

            A_full = np.vstack(rows)
            self.A_cons = sparse.csc_matrix(A_full)

            self.l_template = np.zeros(n_cons)
            self.u_template = np.zeros(n_cons)

            self.solver = osqp.OSQP()
            self.solver.setup(P=self.P, q=np.zeros((self.P.shape[0],)),
                              A=self.A_cons, l=self.l_template, u=self.u_template,
                              verbose=False, warm_start=True)

        q = np.zeros((n_var,))
        for k in range(N):
            q[k*nx:(k+1)*nx] = -self.Q @ x_ref_traj[k]
        q[N*nx:(N+1)*nx] = -self.Q @ x_ref_traj[N]

        need_p_copy = (target_position is not None and position_weight > 0)
        if need_p_copy:
            P_lil = self.P.tolil()
        else:
            P_lil = None

        use_obstacle_slack = False
        points = None
        if enable_obstacle and obstacle_points is not None:
            points = np.asarray(obstacle_points, dtype=np.float64)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 3:
                use_obstacle_slack = True

        if target_position is not None and position_weight > 0:
            target_pos_xy = np.asarray(target_position)
            if P_lil is None:
                P_lil = self.P.tolil()
            for k in range(N):
                base_idx = k * nx
                P_lil[base_idx + 0, base_idx + 0] += 2.0 * position_weight
                P_lil[base_idx + 1, base_idx + 1] += 2.0 * position_weight
                q[base_idx + 0] -= 2.0 * position_weight * target_pos_xy[0]
                q[base_idx + 1] -= 2.0 * position_weight * target_pos_xy[1]

        if P_lil is not None:
            P_cur = P_lil.tocsc()
        else:
            P_cur = self.P

        l = np.zeros((n_cons,))
        u = np.zeros((n_cons,))
        idx = 0

        l[idx:idx+nx] = x0
        u[idx:idx+nx] = x0
        idx += nx

        for k in range(N):
            l[idx:idx+nx] = 0
            u[idx:idx+nx] = 0
            idx += nx

        for k in range(N):
            for i in range(3):
                l[idx] = -self.v_max[i]
                u[idx] = self.v_max[i]
                idx += 1

        for k in range(N):
            for i in range(3):
                l[idx] = -self.a_max[i]
                u[idx] = self.a_max[i]
                idx += 1

        for k in range(N):
            l[idx] = -self.yaw_rate_max
            u[idx] = self.yaw_rate_max
            idx += 1

        if use_obstacle_slack:
            P_cur, q, A_cur, l, u = self._build_obstacle_slack_qp(
                P_cur, q, l, u, self.A_cons, x0, points, N, nx, nu,
            )
        else:
            A_cur = self.A_cons

        self.solver.setup(P=P_cur, q=q, A=A_cur, l=l, u=u,
                          verbose=False, warm_start=True)
        res = self.solver.solve()

        if res.info.status_val in [1, 2]:
            sol = res.x
            n_state = (N + 1) * nx
            n_ctrl = N * nu
            traj = sol[:n_state].reshape((N + 1, nx))
            ctrl = sol[n_state:n_state + n_ctrl].reshape((N, nu))
            return traj, ctrl
        else:
            return None, None
