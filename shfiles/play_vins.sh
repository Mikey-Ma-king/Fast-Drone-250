roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 1
rosbag play $1 -s 2 &
roslaunch vins fast_drone_250.launch &
roslaunch ego_planner single_run_in_exp.launch &
wait
