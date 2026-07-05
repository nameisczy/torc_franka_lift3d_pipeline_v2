#!/usr/bin/env bash
# BAG="recording0003.bag"
BAG="recording0001.bag"
gnome-terminal --tab -- bash -ic "roslaunch lab_vbnpm replay.launch extra:=true bag_file:=$BAG"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm cams_publish.launch"
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm perception.launch"
sleep 3
python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py
