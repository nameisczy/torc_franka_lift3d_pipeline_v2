#!/usr/bin/env bash
FILE_NAME="$1"
rosbag record -O $FILE_NAME \
	/d455/aligned_depth_to_color/image_raw \
	/d455/color/camera_info \
	/d455/color/image_raw \
	/d435/aligned_depth_to_color/image_raw \
	/d435/color/camera_info \
	/d435/color/image_raw \
	/joint_states \
	/joint_states_all
