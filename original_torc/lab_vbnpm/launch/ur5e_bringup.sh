# IP="172.16.90.197"
IP="10.0.0.5"
roslaunch ur_robot_driver ur5e_bringup.launch \
	robot_ip:=$IP \
	kinematics_config:=$(rospack find lab_vbnpm)/launch/calibration/ur5_robot_calibration.yaml
