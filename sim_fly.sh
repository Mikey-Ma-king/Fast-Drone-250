# roslaunch px4 mavros_posix_sitl.launch 
roslaunch px4 mavros_posix_sitl.launch \
  gui:=false \
  verbose:=false \
#   world:=$(rospack find mavlink_sitl_gazebo)/worlds/outdoor1.world \
#   vehicle_sdf:=iris_realsense_camera
# param set COM_RCL_EXCEPT 4
# param set /imu/frequency 200