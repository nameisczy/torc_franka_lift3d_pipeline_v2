#!/bin/bash
for i in "$@"
do
	./docker/set_gpu.sh $i
	NAME=grasp_planning-$i
	./docker/docker_run_ros.sh start $NAME -- trial all --rosbag --mj-pickle --method all_auto --server --rerun-error

	# ./docker/docker_run_ros.sh start $NAME -- trial unstructured --rosbag --mj-pickle --method dg_only --server
	# ./docker/docker_run_ros.sh start $NAME -- trial structured --rosbag --mj-pickle --method dg_only --server --rerun-error
	# ./docker/docker_run_ros.sh start $NAME -- trial unstructured --rosbag --mj-pickle --method dg_into_vlm --server
	# ./docker/docker_run_ros.sh start $NAME -- trial unstructured --rosbag --mj-pickle --method vlm_dg --server
	# ./docker/docker_run_ros.sh start $NAME -- trial structured --rosbag --mj-pickle --method dg_into_vlm --server
	# ./docker/docker_run_ros.sh start $NAME -- trial structured --rosbag --mj-pickle --method vlm_dg --server
	# ./docker/docker_run_ros.sh start $NAME -- trial structured --rosbag --mj-pickle --method all_auto --server
	# ./docker/docker_run_ros.sh start $NAME -- trial unstructured --rosbag --mj-pickle --method all_auto --server
	# ./docker/docker_run_ros.sh start $NAME -- trial difficult --rosbag --mj-pickle --method all_auto --server
done
./docker/set_gpu.sh 0
