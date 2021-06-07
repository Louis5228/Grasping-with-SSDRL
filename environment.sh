#! /bin/bash

source /opt/ros/melodic/setup.bash
source ./catkin_ws/devel/setup.bash

export ROS_IP=127.0.0.1
export ROS_MASTER_URI=http://127.0.0.1:11311

echo "DLP project environment"