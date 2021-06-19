#! /bin/bash

HOME=/home/dlp

export ROS_IP=127.0.0.1
export ROS_MASTER_URI=http://127.0.0.1:11311

source /opt/ros/melodic/setup.bash
source ./catkin_ws/devel/setup.bash

echo "DLP project environment"