# roslaunch px4 mavros_posix_sitl.launch &sleep 4
./px4ctrl.sh &sleep 1
./read.sh &sleep 4
../Fast-Perching/perching.sh &sleep 5
./takeoff.sh &sleep 1
wait