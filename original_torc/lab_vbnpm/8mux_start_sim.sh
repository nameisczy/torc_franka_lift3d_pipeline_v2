#!/bin/bash

DIR="$(rospack find lab_vbnpm)"
SCENE_NAME="shelf_unstructured_105"
# SCENE_NAME="tabletop_unstructured_14_0"
DEFAULT_SCENE="./tests/scenes/adjusted/$SCENE_NAME.xml"
# DEFAULT_SCENE="./tests/scenes/adjusted/shelf_unstructured_122.xml"
SCENE=$(python -c "import os; print(os.path.relpath('${1:-$DEFAULT_SCENE}','$DIR'))")

# FRAMEWORK="human_in_loop"
FRAMEWORK="DG"

OUT_FILE="./experiments/$FRAMEWORK/$SCENE_NAME.csv"

echo "Args: anything or not, control, perception, grasping"

SESSION_NAME="j_user_session"

tmux new-session -d -s $SESSION_NAME

tmux rename-window -t $SESSION_NAME:0 'Control Launch'
tmux send-keys -t $SESSION_NAME:0 "DISABLE_ROS1_EOL_WARNINGS=1 roslaunch lab_vbnpm control.launch is_sim:=true scene:=$SCENE" C-m #ycb_boxes.xml ycb_pile1.xml
z
tmux new-window -t $SESSION_NAME:2 -n 'Container Launch'
tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait cgn_ros container.launch' C-m

sleep 3

# tmux new-window -t $SESSION_NAME:6 -n 'Record rosbag'
# tmux send-keys -t $SESSION_NAME:6 './record_rosbag.sh' C-m

tmux new-window -t $SESSION_NAME:3 -n 'Calibration Simulation Parameters'
tmux send-keys -t $SESSION_NAME:3 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
# tmux send-keys -t $SESSION_NAME:3 'source $(rospack find lab_vbnpm)/API_KEY' C-m

tmux new-window -t $SESSION_NAME:4 -n 'Human in the loop'
tmux send-keys -t $SESSION_NAME:4 "python $(rospack find lab_vbnpm)/scripts/task_planner/curobo_human_in_loop.py s s $OUT_FILE; $(rospack find lab_vbnpm)/7kill.sh" C-m

tmux new-window -t $SESSION_NAME:5 -n 'Selected object name'
tmux send-keys -t $SESSION_NAME:5 'rostopic echo /ground_truth/selected_object_name' C-m

tmux attach-session -t $SESSION_NAME
