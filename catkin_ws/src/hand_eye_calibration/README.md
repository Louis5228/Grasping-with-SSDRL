roslaunch arm_operation ur5_real.launch robot_ip:=192.168.50.11 tool_length:=0.0
roslaunch realsense2_ros rs_rgbd.launch camera:=camera
roslaunch apriltags_ros cali_cam1.launch
roslaunch hand_eye_calibration fake_tcp_publisher.launch
rosrun hand_eye_calibration static_hand_eye_calibration _file_name:=[filename]
rosrun hand_eye_calibration get_transform catkin_ws/src/hand_eye_calibration/data/[filename].txt