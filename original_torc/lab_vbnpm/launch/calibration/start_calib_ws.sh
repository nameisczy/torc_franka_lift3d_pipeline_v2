#!/usr/bin/env bash
gnome-terminal --tab -- bash -ic "roslaunch motoman motoman.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm rs_2_cameras.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm control.launch is_sim:=false moveit:=true"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm cams_publish.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm aruco.launch"

sleep 5
rosservice call /robot_enable "{}"
rosservice call /robot_enable "{}"
