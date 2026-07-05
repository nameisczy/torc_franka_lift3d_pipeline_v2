#!/bin/bash

use_perception=$1
adjust_obj=$2
testing_real=$3

echo "Args: 1st use_perception, 2nd adjust_object, 3rd testing_real. Type anything for an argument to use it. Use the term (not) to say no to an argument."

# Name of the tmux session
SESSION_NAME="j_user_session"

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION_NAME

if [ -z "$testing_real" ]; then
    tmux rename-window -t $SESSION_NAME:0 'Control Launch'
    tmux send-keys -t $SESSION_NAME:0 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_boxes.xml' C-m #ycb_boxes.xml ycb_pile1.xml
else
    tmux rename-window -t $SESSION_NAME:0 'Robot Description'
    tmux send-keys -t $SESSION_NAME:0 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm replay.launch bag_path:=/home/j_user/robot/downloads/recording0002.bag extra:=true' C-m

    #tmux rename-window -t $SESSION_NAME:10 'Robot Description'
    #tmux send-keys -t $SESSION_NAME:10 'source ~/robot/downloads/bot/bin/activate' C-m
    #tmux send-keys -t $SESSION_NAME:10 'roslaunch lab_vbnpm replay.launch bag_path:=/robot/downloads/recording0001.bag' C-m #ycb_boxes.xml ycb_pile1.xml
fi



#if [ -n "$testing_real" ]; then
   # tmux rename-window -t $SESSION_NAME:0 'Just roscore'
  #  tmux send-keys -t $SESSION_NAME:0 'source ~/robot/downloads/bot/bin/activate' C-m
 #   tmux send-keys -t $SESSION_NAME:0 'roscore' C-m
#fi

#Segment3d
#if [ "$use_perception" != "not" ]; then
    #sleep 5
    #tmux new-window -t $SESSION_NAME:1 -n 'Perception Launch'
    #tmux send-keys -t $SESSION_NAME:1 'source ~/robot/downloads/bot/bin/activate' C-m
    #tmux send-keys -t $SESSION_NAME:1 'roslaunch --wait lab_vbnpm perception.launch' C-m
    #sleep 5
#fi

if [ "$use_perception" != "not" ]; then
    sleep 1
    tmux new-window -t $SESSION_NAME:90 -n 'langsam twin server'
    tmux send-keys -t $SESSION_NAME:90 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:90 'python ../segment3d/run_container_sam.py' C-m
    sleep 1
fi

if [ -z "$testing_real" ]; then
    tmux new-window -t $SESSION_NAME:2 -n 'Container2 Launch'
    tmux send-keys -t $SESSION_NAME:2 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait gpd_docker container2.launch' C-m
fi

if [ -n "$adjust_obj" ] && [ "$adjust_obj" != "not" ]; then
    tmux new-window -t $SESSION_NAME:3 -n 'Adjust Object'
    tmux send-keys -t $SESSION_NAME:3 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:3 'python ./scripts/perception/adjust_object.py' C-m
fi

sleep 3

#if [ -z "$testing_real" ]; then
tmux new-window -t $SESSION_NAME:4 -n 'Calibration Simulation Parameters'
tmux send-keys -t $SESSION_NAME:4 'source ~/robot/downloads/bot/bin/activate' C-m
tmux send-keys -t $SESSION_NAME:4 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
#fi

#For when adjust object is true so therefore shouldn't do the gt comparison
if [ -n "$testing_real" ] && [ -n "$adjust_obj" ] && [ "$adjust_obj" != "not" ]; then
    echo "must sleep for 10 before running python closed loop with rosbag as the service needs time to start"
    sleep 10

    #tmux new-window -t $SESSION_NAME:5 -n 'Rosbag Real Images'
    #tmux send-keys -t $SESSION_NAME:5 'rosparam set use_sim_time true' C-m
    #tmux send-keys -t $SESSION_NAME:5 'rosbag play ../../../robot/downloads/recording0001.bag --clock' C-m

    tmux new-window -t $SESSION_NAME:6 -n 'Python Closed Loop'
    tmux send-keys -t $SESSION_NAME:6 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:6 'python scripts/task_planner/closed_loop.py r r "potted meat can" p "nothing" "010_potted_meat_can" True' C-m
fi

if [ -n "$testing_real" ] && [ "$adjust_obj" == "not" ]; then
    echo "must sleep for 10 before running python closed loop with rosbag as the service needs time to start"
    sleep 10

    #tmux new-window -t $SESSION_NAME:5 -n 'Rosbag Real Images'
    #tmux send-keys -t $SESSION_NAME:5 'rosparam set use_sim_time true' C-m
    #tmux send-keys -t $SESSION_NAME:5 'rosbag play ../../../robot/downloads/recording0001.bag --clock' C-m

    tmux new-window -t $SESSION_NAME:6 -n 'Python Closed Loop'
    tmux send-keys -t $SESSION_NAME:6 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:6 'python scripts/task_planner/closed_loop.py r r "potted meat can" p "./xmls/real_experiment_adjusted.xml" "010_potted_meat_can" True' C-m
fi



#tmux new-window -t $SESSION_NAME:4 -n 'Tests'
#tmux send-keys -t $SESSION_NAME:4 'source ~/robot/downloads/bot/bin/activate' C-m
#tmux send-keys -t $SESSION_NAME:4 'roslaunch lab_vbnpm j_user.launch' C-m

#tmux new-window -t $SESSION_NAME:5 -n 'Emboided Segment Anything'
#tmux send-keys -t $SESSION_NAME:5 'source ~/robot/downloads/pointTransformer/bin/activate' C-m
#tmux send-keys -t $SESSION_NAME:5 'python ~/robot/downloads/perception/segment_service.py' C-m

# Optionally add more windows here, repeating the pattern
# tmux new-window -t $SESSION_NAME:N -n 'Window N'
# tmux send-keys -t $SESSION_NAME:N 'your-command-here; bash' C-m

# Attach to the session so you can see the windows and switch between them
tmux attach-session -t $SESSION_NAME
