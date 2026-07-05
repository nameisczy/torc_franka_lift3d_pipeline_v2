#!/usr/bin/env bash
CMD="micromamba activate ros_env && source /home/kchen/Robotics/workspace/devel/setup.bash"
terminator --new-tab -x bash -ic "$CMD && 'roslaunch lab_vbnpm control.launch is_sim:=true scene:=tests/ycb_sorted.xml'"
terminator --new-tab -x bash -ic "'roslaunch --wait lab_vbnpm perception.launch'"

# terminator --new-tab -x bash -ic "'roslaunch graspnet_ros container.launch'"
# terminator --new-tab -x bash -ic "'roslaunch gpg_ros container.launch'"

# CONFIG="$(rospack find lab_vbnpm)/scripts/grasp_planner/gpd_ros_eigen_params.cfg"
# CONFIG="$(rospack find lab_vbnpm)/scripts/grasp_planner/gpd_fast_params.cfg"
# terminator --new-tab -x bash -ic "'roslaunch --wait gpd_docker container.launch config:='$CONFIG"
terminator --new-tab -x bash -ic "'roslaunch --wait gpd_docker container2.launch'"
sleep 3
python $(rospack find lab_vbnpm)/launch/calibration/sim_params.py
