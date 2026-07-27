#!/bin/bash
# sh ./upixels_flow.sh&
# python3 rebort.py
# sleep 1

killall svo_node

source devel/setup.bash
roslaunch px4ctrl run_ctrl.launch&

source ~/schurvins/devel/setup.bash
roslaunch svo_ros euroc_vio_stereo.launch &

sleep 5
source devel/setup.bash
rostopic pub -1  /takeoff_flag quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"

source devel/setup.bash
sh shfiles/takeoff.sh&
sleep 3
stty -F /dev/USB_hc14_send 115200 cs8 -cstopb -parenb
echo "[back_panel] 触发背板序列..."
echo -n "1@" > /dev/USB_hc14_send
wait
