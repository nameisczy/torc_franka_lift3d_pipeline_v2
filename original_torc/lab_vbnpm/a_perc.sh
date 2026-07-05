#!/usr/bin/env bash
CMD="micromamba activate ros_env && source /home/daniel/sauce/robotics/ros/lab_ws/devel/setup.bash"
BAG="tests/recording_0001.bag"
alacritty -e bash -ic "$CMD && roslaunch lab_vbnpm replay.launch bag_file:=$BAG extra:=true planning:=true" &
alacritty -e bash -ic "$CMD && roslaunch --wait lab_vbnpm cams_publish.launch" &
alacritty -e bash -ic "$CMD && roslaunch --wait lab_vbnpm perception.launch" &
