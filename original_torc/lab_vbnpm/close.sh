#!/bin/bash
tmux kill-session -t j_user_session
docker stop $(docker ps -q)
docker rmi -f $(docker images -f "dangling=true" -q)
killall -9 -u j_user