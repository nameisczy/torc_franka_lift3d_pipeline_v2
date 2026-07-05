#!/bin/bash

control_launch=$1
perception=$2
grasping=$3
running_closed_loop=$4

echo "Args: anything or not, control, perception, grasping, closed_loop"

SESSION_NAME="session"
CMD="micromamba activate ros_env && source /data/local/kc1317/workspace/devel/setup.bash"

tmux new-session -d -s $SESSION_NAME

# CONTROL
if [ "$control_launch" == "q" ]; then
    tmux rename-window -t $SESSION_NAME:0 'Control'
    tmux send-keys -t $SESSION_NAME:0 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_02_non_perishables.xml' C-m #ycb_boxes.xml ycb_pile1.xml
    # tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_03_chips_part_occl.xml' C-m #ycb_05_mustard_part_occl.xmlycb_02_non_perishables.xml #ycb_boxes.xml ycb_pile1.xml ycb_01_boxes.xml
    # tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_06_crackers_part_occl.xml' C-m #ycb_05_mustard_part_occl.xmlycb_02_non_perishables.xml #ycb_boxes.xml ycb_pile1.xml ycb_01_boxes.xml
    # tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_05_mustard_part_occl.xml' C-m #ycb_05_mustard_part_occl.xmlycb_02_non_perishables.xml #ycb_boxes.xml ycb_pile1.xml ycb_01_boxes.xml
elif [ "$control_launch" == "rosbag" ]; then
    tmux rename-window -t $SESSION_NAME:0 'Robot Description'
    tmux send-keys -t $SESSION_NAME:0 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:0 'roslaunch lab_vbnpm replay.launch bag_path:=/home/j_user/robot/downloads/02_non-perishables.bag extra:=true planning:=true' C-m #recording0002.bag
fi

# PERCEPTION
if [ "$perception" != "not" ] && [ "$perception" != "test" ] && [ "$perception" != "detic" ]; then
    sleep 1
    tmux new-window -t $SESSION_NAME:90 -n 'langsam twin server'
    tmux send-keys -t $SESSION_NAME:90 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:90 'python src/segment3d/run_container_sam.py' C-m
    sleep 1
elif [ "$perception" == "test" ]; then
    tmux new-window -t $SESSION_NAME:77 -n 'gsam'
    tmux send-keys -t $SESSION_NAME:77 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:77 'python src/segment3d/run_container_gsam.py' C-m

    sleep 5

    tmux new-window -t $SESSION_NAME:78 -n 'client'
    tmux send-keys -t $SESSION_NAME:78 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:78 'python src/segment3d/src/gsam_service.py' C-m
elif [ "$perception" == "detic" ]; then
    tmux new-window -t $SESSION_NAME:3 -n 'perception'
    tmux send-keys -t $SESSION_NAME:3 "$CMD" C-m
    # tmux send-keys -t $SESSION_NAME:3 'roslaunch --wait lab_vbnpm perception_detic.launch' C-m
    tmux send-keys -t $SESSION_NAME:3 'roslaunch --wait lab_vbnpm perception.launch' C-m
elif [ "$perception" == "not" ]; then
    sleep 1
fi

# GRASPING
if [ "$grasping" != "not" ]; then
    tmux new-window -t $SESSION_NAME:2 -n 'GPD'
    tmux send-keys -t $SESSION_NAME:2 "$CMD" C-m
    # tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait gpd_docker container2.launch' C-m
    tmux send-keys -t $SESSION_NAME:2 'roslaunch --wait cgn_ros container.launch' C-m
fi

sleep 3

# if [ "$control_launch" != "rosbag" ]; then
#     tmux new-window -t $SESSION_NAME:4 -n 'Script'
#     tmux send-keys -t $SESSION_NAME:4 "$CMD" C-m
#     tmux send-keys -t $SESSION_NAME:4 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m

#     sleep 20
#     tmux send-keys -t $SESSION_NAME:4 'python src/lab_vbnpm/tests/curobo_open_loop.py s g 35' C-m
# fi

# SCRIPT
if [ "$running_closed_loop" != "not" ]; then
    sleep 3
    tmux new-window -t $SESSION_NAME:7 -n 'Closed-loop'
    #tmux send-keys -t $SESSION_NAME:7 'source ~/robot/downloads/bot/bin/activate' C-m
    # tmux send-keys -t $SESSION_NAME:7 'deactivate' C-m
    # tmux send-keys -t $SESSION_NAME:7 'conda activate cur' C-m
    tmux send-keys -t $SESSION_NAME:7 "$CMD" C-m
    tmux send-keys -t $SESSION_NAME:7 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
    sleep 20
    if [ "$running_closed_loop" == "real" ]; then
        #sleep 20
        #tmux send-keys -t $SESSION_NAME:7 'python scripts/task_planner/closed_loop.py s r "white box." p' C-m
        tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/curobo_closed_loop.py s r "red box." p' C-m
    elif [ "$running_closed_loop" == "ag" ]; then
        tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/curobo_active_grasp.py s g 35' C-m
    elif [ "$running_closed_loop" == "gt" ]; then
        tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/curobo_closed_loop.py s g 35 p' C-m
    else
        tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/curobo_closed_loop.py s g 35 p "nothing" "010_potted_meat_can" True' C-m 
    fi
else
    tmux new-window -t $SESSION_NAME:7 -n 'Open-loop'
    tmux send-keys -t $SESSION_NAME:7 "$CMD" C-m
    sleep 20
    tmux send-keys -t $SESSION_NAME:7 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m
    tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/tests/curobo_open_loop.py s g 35' C-m
fi

tmux new-window -t $SESSION_NAME:10 -n 'Input'

# if [ "$running_closed_loop" != "not" ]; then
#     tmux new-window -t $SESSION_NAME:7 -n 'Python Closed Loop'
#     tmux send-keys -t $SESSION_NAME:7 "$CMD" C-m
#     tmux send-keys -t $SESSION_NAME:7 'python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py' C-m

#     if [ "$running_closed_loop" == "real" ]; then
#         sleep 20
#         tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/closed_loop.py s r "white box." p' C-m
#     else
#         tmux send-keys -t $SESSION_NAME:7 'python src/lab_vbnpm/scripts/task_planner/closed_loop.py s g 35 p' C-m
#     fi
# fi

# if [ "$control_launch" == "rosbag" ]; then
#     tmux new-window -t $SESSION_NAME:600 -n 'Python Closed Loop'
#     tmux send-keys -t $SESSION_NAME:600 "$CMD" C-m
#     tmux send-keys -t $SESSION_NAME:600 'python src/lab_vbnpm/scripts/task_planner/closed_loop.py r r "mustard." p "nothing" "010_potted_meat_can" True' C-m
# fi

tmux attach-session -t $SESSION_NAME
