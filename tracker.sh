#!/bin/bash
source devel/setup.bash
roslaunch planning traj_server.launch

rosbag record  /mavros/setpoint_raw/attitude  /vins_fusion/imu_propagate /position_cmd /target_ekf_odom /flow_data /px4ctrl/takeoff_land /dog_pos_processed /dog_pos /beacon /uwb/odometry/a3 /AOA_Tag_data /land_mark /test_mark /drone2/planning/traj /traj_v -o /home/pc/perching_bag/perching
wait
