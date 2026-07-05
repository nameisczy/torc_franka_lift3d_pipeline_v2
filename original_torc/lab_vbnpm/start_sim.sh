#!/usr/bin/env bash
DIR="$(rospack find lab_vbnpm)"
# SCENE="tests/TEST.xml"
# SCENE="tests/ycb_01_boxes.xml"
# SCENE="tests/ycb_02_non_perishables.xml"
# SCENE="tests/ycb_03_chips_part_occl.xml"
# SCENE="tests/ycb_04_pudding_part_occl.xml"
# SCENE="tests/ycb_05_mustard_part_occl.xml"
SCENE="tests/ycb_06_crackers_part_occl.xml"
# SCENE="tests/ycb_07_soup_can_tot_occl.xml"
# SCENE="tests/ycb_08_chips_tot_occl.xml"
# SCENE="tests/ycb_09_mustard_tot_occl.xml"
# SCENE="tests/ycb_clutter.xml"
# SCENE="tests/ycb_fruit.xml"
ARG="$1"
[ -z "$ARG" ] || SCENE=$(python -c "import os; print(os.path.relpath('$ARG','$DIR'))")

gnome-terminal --tab -- bash -ic "roslaunch lab_vbnpm control.launch is_sim:=true scene:=$SCENE" &
gnome-terminal --tab -- bash -ic "roslaunch --wait lab_vbnpm perception.launch client_only:=true" &

# gnome-terminal --tab -- bash -ic "roslaunch graspnet_ros container.launch" &
# gnome-terminal --tab -- bash -ic "roslaunch gpg_ros container.launch" &

# CONFIG="$(rospack find lab_vbnpm)/scripts/grasp_planner/gpd_ros_eigen_params.cfg"
# gnome-terminal --tab -- bash -ic "roslaunch --wait gpd_docker container.launch config:=$CONFIG" &

# gnome-terminal --tab -- bash -ic "roslaunch --wait gpd_docker container2.launch" &
# gnome-terminal --tab -- bash -ic "roslaunch --wait cgn_ros container.launch" &
sleep 3
python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py
