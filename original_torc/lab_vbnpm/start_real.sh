#!/usr/bin/env bash
gnome-terminal --tab -- bash -ic "roslaunch motoman motoman.launch"
# gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm rs_2_cameras.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm zed_m.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm control.launch is_sim:=false"
# gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm control.launch is_sim:=false moveit:=true"

# rosrun dynamic_reconfigure dynparam set /d455/stereo_module "{'visual_preset': 3}"
# rosrun dynamic_reconfigure dynparam set /d435/stereo_module "{'visual_preset': 3}"

gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm cams_publish.launch"
# gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm perception.launch client_only:=true"
#
# gnome-terminal --tab -- bash -ic "roslaunch --wait graspnet_ros container.launch"
# gnome-terminal --tab -- bash -ic "roslaunch --wait gpg_ros container.launch"

# CONFIG="$(rospack find lab_vbnpm)/scripts/grasp_planner/gpd_ros_eigen_params.cfg"
# CONFIG="$(rospack find lab_vbnpm)/scripts/grasp_planner/gpd_ros_large_bite.cfg"
# gnome-terminal --tab -- bash -ic "roslaunch --wait gpd_docker container.launch config:=$CONFIG"
# gnome-terminal --tab -- bash -ic "roslaunch --wait gpd_docker container2.launch"
sleep 3
python $(rospack find lab_vbnpm)/launch/calibration/set_workspace_params.py
rosservice call /robot_enable "{}"
rosservice call /robot_enable "{}"
