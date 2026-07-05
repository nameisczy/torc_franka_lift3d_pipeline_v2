#!/bin/bash

print_help() {
	echo "Usage: $0 [start|resume|stop]"
	echo
	echo "Commands:"
	echo "  start   Start docker trials"
	echo "  resume  Resume docker trials using resume_trial mode"
	echo "  stop    Stop all docker trial containers"
}

if [ $# -eq 0 ]; then
	print_help
	exit 0
fi

CMD="$1"

case "$CMD" in
	-h|--help|help)
		print_help
		exit 0
		;;
	start|resume|stop)
		;;
	*)
		print_help
		exit 1
		;;
esac

TRIALS=(
	# "all:0:0:all_auto"
	# "all:1:1:all_auto"
	# "all:2:2:all_auto"
	# "all:3:3:all_auto"
	# "all:4:4:all_auto"
	# "all:5:5:all_auto"
	# "all:6:0:all_auto"
	# "all:7:1:all_auto"
	"all:6:a6:all_auto"
	"all:7:a7:all_auto"
)
if [ "$CMD" = "start" ]; then
	echo -e "\n\e[1;32mStarting trials...\e[0m"
elif [ "$CMD" = "resume" ]; then
	echo -e "\n\e[1;32mResuming trials...\e[0m"
else
	echo -e "\n\e[1;32mStopping trials...\e[0m"
fi
for TRIAL in "${TRIALS[@]}"
do
	IFS=":" read -r TRIAL_CONFIG GPU_ID CONTAINER_ID METHOD <<< "$TRIAL"
	NAME=grasp_planning-$CONTAINER_ID
	echo "  $NAME (GPU $GPU_ID): $METHOD: $TRIAL_CONFIG"

	if [ "$CMD" = "start" ]; then
		./docker/set_gpu.sh $GPU_ID
		# Start trial
		./docker/docker_run_ros.sh start $NAME -- trial $TRIAL_CONFIG --rosbag --mj-pickle --method $METHOD --server --rerun-error
	elif [ "$CMD" = "resume" ]; then
		./docker/set_gpu.sh $GPU_ID
		# Resume trial
		./docker/docker_run_ros.sh start $NAME -- resume_trial --server "experiments/runs/$NAME/trial*" --rerun-error
	else
		# Stop trial
		./docker/docker_run_ros.sh stop $NAME
	fi
done
if [ "$CMD" != "stop" ]; then
	./docker/set_gpu.sh 0
fi
