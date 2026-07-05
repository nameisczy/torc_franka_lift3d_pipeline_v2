#!/usr/bin/bash
source ../../devel/setup.bash
BAG_FILE_PATH="$1"
./killall.sh
export DISABLE_ROS1_EOL_WARNINGS=1
roslaunch lab_vbnpm replay.launch bag_file:=$BAG_FILE_PATH
