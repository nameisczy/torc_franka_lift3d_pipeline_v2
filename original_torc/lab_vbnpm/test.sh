#!/bin/bash

# Define the session name
SESSION_NAME="my_trap_session"

# Check if the tmux session already exists and kill it if it does
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# Define a function to be executed by the new tmux session
function pane_script() {
    # This is the actual code that will run inside the tmux pane
    cleanup_function() {
        touch /tmp/trap_caught
        exit 0
    }

    # Trap the termination signals
    trap cleanup_function SIGINT SIGTERM SIGHUP

    echo 'Script is running in a tmux pane...'
    echo 'Press Ctrl+C to trigger the trap and exit.'
    echo 'Waiting for a signal...'

    # Keep the script running until a signal is received
    sleep infinity
}

# Export the function so it can be called by the new tmux session
export -f pane_script

# Create a new tmux session and run the exported function
echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" "bash -c 'pane_script'"

echo "Session '$SESSION_NAME' started."
echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
echo "To kill the session from outside, run: tmux kill-session -t $SESSION_NAME"