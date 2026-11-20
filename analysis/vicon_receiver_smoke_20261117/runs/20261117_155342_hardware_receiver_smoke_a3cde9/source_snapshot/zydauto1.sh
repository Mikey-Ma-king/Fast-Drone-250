#!/bin/bash
source devel/setup.bash
sudo chmod 777 /dev/ttyACM0
roslaunch realsense2_camera rs_camera.launch &
# 独立评价接收端；不重映射到 SchurVINS/控制器输入，不发布 TF。
roslaunch mocap_vicon vicon.launch &
sleep 2

# ./send+.sh&

sleep 2
./tracker.sh &

./read.sh&
./MPC.sh &
# python3 dog_pos_processor.py &
# roslaunch planning dog_pos_processor.launch &
# 新版 Ours-SW3 融合：启用下一行时，注释上一行；切换前先停止当前 processor。
roslaunch planning dog_pos_processor_new.launch &
roslaunch mavros px4.launch &
wait
# roslaunch vins fast_drone_250.launch &
