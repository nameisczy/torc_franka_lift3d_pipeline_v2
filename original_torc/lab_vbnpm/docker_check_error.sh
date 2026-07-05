#!/usr/bin/env bash
PANE="${1:-2}"
LN="${2:-5}"

for container in $(docker ps -f name="$FULL_CONTAINER_NAME" --format '{{.Names}}')
do
	if [[ "$container" == *"vbnpm_ros"* ]]
	then
		echo -e "$container:\n"
		docker exec $container tmux capture-pane -t %$PANE -S -20 -p | grep . | tail -n "$LN"
		echo
		echo
	fi
done
