#!/bin/bash

./run_experiment.sh trial --method vlm_dg unstructured
./run_experiment.sh trial --method vlm_dg structured

# SESSION_NAME="run_experiment"

# echo "Creating new tmux session: $SESSION_NAME"
# tmux new-session -d -s "$SESSION_NAME"

# tmux send-keys -t "$SESSION_NAME:0.0" "./run_experiment.sh trial --method vlm_dg unstructured" C-m
# tmux send-keys -t "$SESSION_NAME:0.0" "./run_experiment.sh trial --method vlm_dg structured" C-m

# tmux attach-session -t "$S[qESSION_NAME"