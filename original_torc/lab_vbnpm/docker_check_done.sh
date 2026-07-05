#!/usr/bin/env bash
PANE="${1:-0}"
LN="${2:-5}"

echo
for container in $(docker ps -f name="$FULL_CONTAINER_NAME" --format '{{.Names}}')
do
	if [[ "$container" == *"vbnpm_ros"* ]]
	then
		echo -e "\033[1m$container:\033[0m\n"
		docker exec $container tmux capture-pane -t docker-runner:$PANE -S -50 -p | grep . | tail -n "$LN"
		echo
		echo
	fi
done
