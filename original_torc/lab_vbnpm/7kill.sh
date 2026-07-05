#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$ROOT_DIR/kill_docker.sh
$ROOT_DIR/killall.sh

WINDOWS=("Selected Info" "Control Terminal")
SESSIONS=("j_user_session" "control_sim" "control_sim_headless" "control_sim_server" "control_real" "control_real_headless")

# This kills VSCode on linux for some reason when running run_ros???
# for window in "${WINDOWS[@]}"; do
#     echo "killing window: $window"
#     wmctrl -c $window
# done

for session in "${SESSIONS[@]}"; do
    tmux kill-session -t "$session"
done