#!/usr/bin/env bash
gnome-terminal --tab -- bash -ic "roslaunch motoman motoman.launch"
# gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm rs_2_cameras.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm zed_m.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm control.launch is_sim:=false moveit:=true"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm checkerboard.launch"
# gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm camera_arm_calibrate.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm camera_torso_calibrate.launch"

sleep 3
rosservice call /robot_enable "{}"
rosservice call /robot_enable "{}"
