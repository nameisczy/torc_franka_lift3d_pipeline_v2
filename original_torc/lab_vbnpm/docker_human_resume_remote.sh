#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_COMPOSE_FILE="$DIR/docker/docker-compose-hil.yaml"
export NAME="human_in_the_loop"
# ./docker/docker_run_ros.sh start $NAME -- experiment --scene tests/scenes/final/unstructured_30.xml --target-object obj_000043_0 --method human --server
./docker/docker_run_ros.sh start $NAME -- resume_trial --server "$1"
docker exec -i $NAME-vbnpm_ros-1 bash -c 'cat > ~/.Xauthority' < ~/.Xauthority
./docker/docker_run_ros.sh attach $NAME vbnpm_ros
