sleep 4
rosservice call /mavros/set_message_interval 31 200;
rosservice call /mavros/set_message_interval 105 200;
./px4ctrl.sh &sleep 1
./read.sh &sleep 4
../Fast-Perching/perching.sh &sleep 5
./takeoff.sh &sleep 1
wait