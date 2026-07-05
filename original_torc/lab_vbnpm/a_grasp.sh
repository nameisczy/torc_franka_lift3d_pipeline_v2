#!/usr/bin/env bash
CMD="micromamba activate ros_env && source /home/daniel/sauce/robotics/ros/lab_ws/devel/setup.bash"
# BAG="tests/recording_0002.bag"
BAG=""
if [ -z "$BAG" ]
then
	alacritty -e bash -ic "$CMD && roslaunch lab_vbnpm replay.launch extra:=true planning:=true" &
else
	alacritty -e bash -ic "$CMD && roslaunch lab_vbnpm replay.launch bag_file:=$BAG extra:=true planning:=true" &
	alacritty -e bash -ic "$CMD && roslaunch --wait lab_vbnpm cams_publish.launch" &
	alacritty -e bash -ic "$CMD && roslaunch --wait lab_vbnpm perception.launch" &
fi
alacritty -e bash -ic "$CMD && roslaunch --wait gpd_docker container2.launch" &
# sleep 3
# python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py
