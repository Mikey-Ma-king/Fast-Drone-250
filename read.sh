#!/bin/bash
sudo chmod 777 /dev/video0
sudo chmod 777 /dev/video1
sudo chmod 777 /dev/video2
sudo chmod 777 /dev/video3
sudo chmod 777 /dev/video4
sudo chmod 777 /dev/video5
sudo chmod 777 /dev/video6
# sudo chmod 777 /dev/video7
# sudo chmod 777 /dev/video8
source devel/setup.bash

roslaunch read perching.launch