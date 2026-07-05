#!/usr/bin/env bash
for i in "$@"
do
	./docker/set_gpu.sh $i
	NAME=grasp_planning-$i
	# ./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial*" --rerun-error
	# ./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial*__unstructured__*" --rerun-error
	# ./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial*__structured__*" --rerun-error
	./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial*" --rerun-error
	# ./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial_2026-02-23_03*" --rerun-error
done
./docker/set_gpu.sh 0
