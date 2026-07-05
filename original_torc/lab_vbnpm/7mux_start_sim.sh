#!/bin/bash

DIR="$(rospack find lab_vbnpm)"
DEFAULT_SCENE="./tests/scenes/adjusted/shelf_structured_130.xml"
SCENE=$(python -c "import os; print(os.path.relpath('${1:-$DEFAULT_SCENE}','$DIR'))")

echo "Args: anything or not, control, perception, grasping"

SESSION_NAME="j_user_session"

tmux new-session -d -s $SESSION_NAME

tmux rename-window -t $SESSION_NAME:0 'Control Launch'
tmux send-keys -t $SESSION_NAME:0 "DISABLE_ROS1_EOL_WARNINGS=1 roslaunch lab_vbnpm control.launch is_sim:=true scene:=$SCENE" C-m #ycb_boxes.xml ycb_pile1.xml

tmux new-window -t $SESSION_NAME:2 -n 'Container Launch'
tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait cgn_ros container.launch' C-m

sleep 3

tmux new-window -t $SESSION_NAME:3 -n 'Calibration Simulation Parameters'
tmux send-keys -t $SESSION_NAME:3 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
# tmux send-keys -t $SESSION_NAME:3 'source $(rospack find lab_vbnpm)/API_KEY' C-m

tmux new-window -t $SESSION_NAME:4 -n 'Human in the loop'
tmux send-keys -t $SESSION_NAME:4 'python $(rospack find lab_vbnpm)/scripts/task_planner/curobo_runner.py' C-m

tmux attach-session -t $SESSION_NAME
