#!/bin/bash

# Define the tmux session name.
SESSION_NAME="run_experiment"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the command and arguments in an array.
# This correctly passes each argument as a separate item.
RUN_CMD=( python3 "$ROOT_DIR/experiments/run_experiment.py" "$@" )

# Check if the tmux session already exists.
tmux has-session -t "$SESSION_NAME" 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME"
    
    tmux send-keys -t "$SESSION_NAME:0.0" "export PYTHONPATH=\"\$PYTHONPATH:$ROOT_DIR\" && cd '$ROOT_DIR' && source ../../devel/setup.sh" C-m
    
    # Use printf with the array to correctly pass arguments.
    # The 'tmux send-keys' command needs the arguments as a string.
    # So we join the array elements with spaces.
    printf -v RUN_CMD_STR '%s ' "${RUN_CMD[@]}"
    tmux send-keys -t "$SESSION_NAME:0.0" "$RUN_CMD_STR" C-m
    
    echo "Experiment started in detached tmux session."
else
    echo "Session '$SESSION_NAME' already exists. Attaching..."
fi

# Attach to the tmux session.
tmux attach-session -t "$SESSION_NAME"