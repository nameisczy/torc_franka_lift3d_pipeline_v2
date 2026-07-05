#!/usr/bin/env bash
#
# run_all.sh — launch one tmux window for each combination of (s|t) × (s|u)

# tmux allows script to run asynchronously, even when you close your ilab session

SESSION="validate_scenes"

# 1) start a new detached session
tmux new-session -d -s "$SESSION" -n "launcher"

# 2) iterate all combos
for surface in s t; do
  for structure in s u; do
    WIN="${surface}_${structure}"
    # create a new window
    tmux new-window -t "$SESSION" -n "$WIN"
    tmux send-keys -t "micromamba activate env2" C-m
    tmux send-keys -t "source devel/setup.bash" C-m
    tmux send-keys -t "source /data/local/kc1317/graspclutter6d_mujoco_sim/GC6D_ROOT.sh" C-m
    # send the command into it (replace ./your_script.sh with your script path)
    tmux send-keys -t "$SESSION":"$WIN" \
      "./src/lab_vbnpm/tests/scenes/scene_validation.sh $surface $structure" C-m
  done
done

# 3) (optional) kill the initial launcher window
tmux kill-window -t "$SESSION":"launcher"

# 4) attach to the session
tmux attach-session -t "$SESSION"
