#!/bin/bash

control_launch=$1
perception=$2
grasping=$3
running_closed_loop=$4

echo "Args: anything or not, control, perception, grasping, closed_loop. Try: q test not real"

SESSION_NAME="j_user_session"

tmux new-session -d -s $SESSION_NAME

#CONTROL
if [ "$control_launch" == "q" ]; then
    tmux rename-window -t $SESSION_NAME:0 'Control Launch'
    tmux send-keys -t $SESSION_NAME:0 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_04_pudding_part_occl.xml' C-m #ycb_04_crackers_part_occl.xmlycb_05_mustard_part_occl.xmlycb_02_non_perishables.xml #ycb_boxes.xml ycb_pile1.xml ycb_01_boxes.xmlycb_08_chips_tot_occl.xml
elif [ "$control_launch" == "rosbag" ]; then
    tmux rename-window -t $SESSION_NAME:0 'Robot Description'
    tmux send-keys -t $SESSION_NAME:0 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm replay.launch bag_path:=/home/j_user/robot/downloads/recording0005.bag extra:=true planning:=true' C-m #recording0002.bag, 02_non-perishables.bag
fi

#PERCEPTION
if [ "$perception" != "not" ] && [ "$perception" != "test" ]; then
    sleep 1
    tmux new-window -t $SESSION_NAME:90 -n 'langsam twin server'
    tmux send-keys -t $SESSION_NAME:90 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:90 'python ../segment3d/run_container_sam.py' C-m
    sleep 1
elif [ "$perception" == "test" ]; then
    tmux new-window -t $SESSION_NAME:77 -n 'gsam itself'
    tmux send-keys -t $SESSION_NAME:77 'python ~/catkin_ws/src/segment3d/run_container_gsam.py 0' C-m

    sleep 5

    tmux new-window -t $SESSION_NAME:78 -n 'gsam service'
    tmux send-keys -t $SESSION_NAME:78 'python ~/catkin_ws/src/segment3d/src/gsam_service.py' C-m
fi

#GRASPING
if [ "$grasping" != "not" ]; then
    tmux new-window -t $SESSION_NAME:2 -n 'Container2 Launch'
    tmux send-keys -t $SESSION_NAME:2 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait gpd_docker container2.launch' C-m
    #tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait cgn_ros container.launch' C-m
fi

sleep 3

#CALIBRATION
if [ "$control_launch" != "rosbag" ]; then
    tmux new-window -t $SESSION_NAME:4 -n 'Calibration Simulation Parameters'
    tmux send-keys -t $SESSION_NAME:4 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:4 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
fi

#CLOSED LOOP
if [ "$running_closed_loop" != "not" ]; then
    sleep 3
    tmux new-window -t $SESSION_NAME:7 -n 'Python Closed Loop'
    #tmux send-keys -t $SESSION_NAME:7 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:7 'deactivate' C-m
    tmux send-keys -t $SESSION_NAME:7 'conda activate cur' C-m
    if [ "$running_closed_loop" == "real" ]; then
        #sleep 20
        #tmux send-keys -t $SESSION_NAME:7 'python scripts/task_planner/closed_loop.py s r "white box." p' C-m
        tmux send-keys -t $SESSION_NAME:7 'python scripts/task_planner/curobo_closed_loop.py s r "red box." p' C-m
    elif [ "$running_closed_loop" == "gt" ]; then
        tmux send-keys -t $SESSION_NAME:7 'python scripts/task_planner/curobo_closed_loop.py s g 35 p' C-m
    elif [ "$running_closed_loop" == "evaluation" ]; then
        tmux send-keys -t $SESSION_NAME:7 'python tests/motion_planning_cl_test.py' C-m
    else
        tmux send-keys -t $SESSION_NAME:7 'python scripts/task_planner/curobo_closed_loop.py s g 35 p "nothing" "010_potted_meat_can" True' C-m 
    fi
fi

if [ "$control_launch" == "rosbag" ]; then
    tmux new-window -t $SESSION_NAME:600 -n 'Python Closed Loop'
    tmux send-keys -t $SESSION_NAME:600 'source ~/robot/downloads/bot/bin/activate' C-m
    tmux send-keys -t $SESSION_NAME:600 'python scripts/task_planner/closed_loop.py r r "yellow box." p "nothing" "010_potted_meat_can" True' C-m
fi

sleep 5

tmux new-window -t $SESSION_NAME:123 -n 'Perception pipeline output'
tmux send-keys -t $SESSION_NAME:123 'source ~/robot/downloads/bot/bin/activate' C-m
tmux send-keys -t $SESSION_NAME:123 'tail -f perception_pipeline.log' C-m

tmux new-window -t $SESSION_NAME:125 -n 'Grasping pipeline output'
tmux send-keys -t $SESSION_NAME:125 'source ~/robot/downloads/bot/bin/activate' C-m
tmux send-keys -t $SESSION_NAME:125 'tail -f grasping_pipeline.log' C-m

tmux attach-session -t $SESSION_NAME
