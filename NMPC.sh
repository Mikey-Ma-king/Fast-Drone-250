#!/bin/bash
source /home/pc/Fast-Drone-250/devel/setup.bash

pkill -9 -f "python3 NMPC.py"

# 启动 Python 脚本（后台）
python3 NMPC.py &
MPC_PID=$!

# rosbag record  /mavros/setpoint_raw/attitude  /vins_fusion/imu_propagate /position_cmd /target_ekf_odom /flow_data /px4ctrl/takeoff_land /dog_pos /beacon /uwb/odometry/a3 /AOA_Tag_data /land_mark /drone2/planning/traj -o /home/pc/perching_bag/perching
trap "kill -9 $MPC_PID" SIGINT

wait
# /drone0/planning/traj /drone0/planning/visible_region
