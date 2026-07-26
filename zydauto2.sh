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
echo "[back_panel] 抬起背板..."
./send+.sh &
sleep 2
echo "[back_panel] 放下背板..."
./send-.sh &
wait
