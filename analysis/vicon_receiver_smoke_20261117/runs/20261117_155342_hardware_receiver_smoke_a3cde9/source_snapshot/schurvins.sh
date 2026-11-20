#!/bin/bash
source devel/setup.bash
sudo chmod 777 /dev/ttyACM0
roslaunch mavros px4.launch &
sleep 2
roslaunch realsense2_camera rs_camera.launch &
sleep 2

source ~/schurvins/devel/setup.bash
roslaunch svo_ros euroc_vio_stereo.launch &
wait

# sleep 2

# python3 schurvins.py