from MPC_python_complie import DroneMPC
import numpy as np
import time

def predict_target_trajectory(pos, vel, N, dt):
    return np.array([np.concatenate([pos + vel * t * dt, vel]) for t in range(N+1)])

start_time = time.time()
for i in range(1):

    target_p =np.array([10.0,1.0,0.0])
    target_v =np.array([0.5,0.0,0.0])
    target_yaw = 0.0
    triger = 0
    vins_p =np.array([1.0,0.0,0.0])
    vins_v =np.array([0.0,0.0,0.0])
    vins_yaw = 0

    drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])

    mpc = DroneMPC()

    target_traj = predict_target_trajectory(target_p, target_v, mpc.N, mpc.dt)
    new_x_opt, new_u_opt = mpc.solve(drone_state, target_traj)
    print(new_x_opt)

print("time:",time.time() - start_time)
# print(new_x_opt)