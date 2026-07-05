#!/usr/bin/env bash
# NOTE: The line below kills VSCode for some reason
pkill -ef -9 multiprocessing
docker ps -q
[ -z "$(docker ps -q)" ] || docker kill $(docker ps -q)
rosnode kill -a
killall -wg roslaunch
killall -wg rosmaster
killall -wg rviz
