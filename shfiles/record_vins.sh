sudo chmod 777 /dev/ttyACM0
roslaunch mavros px4.launch &
sleep 2
roslaunch realsense2_camera rs_camera.launch &
rosbag record --tcpnodelay \
/camera/infra1/image_rect_raw \
/camera/infra2/image_rect_raw \
/camera/color/image_raw \
/camera/depth/image_rect_raw \
/mavros/imu/data_raw
wait
