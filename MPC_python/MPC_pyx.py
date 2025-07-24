# cython: language_level=3

import numpy as np
cimport numpy as np

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
        self.yaw_rate_max = np.radians(30)  # max yaw rate: 30 degrees/s

        self.P = None
        self.A_cons = None
        self.solver = None

    def solve(self, np.ndarray[np.float64_t, ndim=1] x0, np.ndarray[np.float64_t, ndim=2] x_ref_traj):
        cdef int nx = self.nx
        cdef int nu = self.nu
        cdef int N = self.N
        cdef int n_var = (N+1) * nx + N * nu

        cdef int n_init_cons = nx
        cdef int n_dyn_cons = N * nx
        cdef int n_vel_cons = N * 3
        cdef int n_acc_cons = N * 3
        cdef int n_yaw_cons = N
        cdef int n_cons = n_init_cons + n_dyn_cons + n_vel_cons + n_acc_cons + n_yaw_cons

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

        self.solver.update(q=q, l=l, u=u)
        res = self.solver.solve()

        if res.info.status_val in [1, 2]:  # solved or solved_inaccurate
            sol = res.x
            traj = sol[:(N+1)*nx].reshape((N+1, nx))
            ctrl = sol[(N+1)*nx:].reshape((N, nu))
            return traj, ctrl
        else:
            return None, None
