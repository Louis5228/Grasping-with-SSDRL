#! /bin/bash

touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_gripper_control/CATKIN_IGNORE
touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_gripper_joint_state_publisher/CATKIN_IGNORE
touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_rviz/CATKIN_IGNORE
touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_gripper_articulated_msgs/CATKIN_IGNORE
touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_gripper_articulated_gazebo/CATKIN_IGNORE
touch ./catkin_ws/src/ur5/robotiq/robotiq_3f_gripper_articulated_gazebo_plugins/CATKIN_IGNORE

catkin_make -C ./catkin_ws
