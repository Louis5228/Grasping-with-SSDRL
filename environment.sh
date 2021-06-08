#! /bin/bash

HOME=/home/dlp

source /opt/ros/melodic/setup.bash
source $HOME/.bashrc
source ./catkin_ws/devel/setup.bash
load_pyrobot_env

export ROS_IP=127.0.0.1
export ROS_MASTER_URI=http://127.0.0.1:11311

echo "DLP project environment"