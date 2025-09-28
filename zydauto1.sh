#!/bin/bash
source devel/setup.bash
sudo chmod 777 /dev/ttyACM0
roslaunch realsense2_camera rs_camera.launch &
sleep 2

# ./send+.sh&

sleep 2
./tracker.sh &

./read.sh&
# ./MPC.sh &
python3 dog_pos_processor.py &
roslaunch mavros px4.launch &
wait
# roslaunch vins fast_drone_250.launch &