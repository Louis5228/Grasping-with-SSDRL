#! /bin/bash

HOME=/home/dlp

export ROS_MASTER_URI=http://140.113.148.82:11311

if [ "$1" ]; then
    echo "ROS IP $1"
    export ROS_IP=$1
else
    echo "ROS IP 127.0.0.1"
    export ROS_IP=127.0.0.1
fi

if [ "$2" ]; then
    echo "ROS MASRER $2"
    export ROS_MASTER_URI=http://$2:11311
else
    echo "ROS MASRER 127.0.0.1"
    export ROS_MASTER_URI=http://127.0.0.1:11311
fi

source /opt/ros/melodic/setup.bash
source ./catkin_ws/devel/setup.bash

echo "DLP project environment"
