cd Fast-Drone-250/ ;
source ~/Fast-Drone-250/devel/setup.bash ;
sudo chmod 777 /dev/ttyACM0 &
roslaunch realsense2_camera rs_camera.launch &
sleep 2
roslaunch mavros px4.launch &
sleep 2
cd ~ ;
source ~/schurvins/devel/setup.bash ;
roslaunch svo_ros euroc_vio_stereo.launch &
wait
