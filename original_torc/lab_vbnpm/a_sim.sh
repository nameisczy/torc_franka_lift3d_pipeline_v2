#!/usr/bin/env bash
DIR="$(rospack find lab_vbnpm)"
# SCENE="tests/TEST.xml"
SCENE="tests/ycb_01_boxes.xml"
# SCENE="tests/ycb_02_non_perishables.xml"
# SCENE="tests/ycb_03_chips_part_occl.xml"
# SCENE="tests/ycb_04_pudding_part_occl.xml"
# SCENE="tests/ycb_05_mustard_part_occl.xml"
# SCENE="tests/ycb_06_crackers_part_occl.xml"
# SCENE="tests/ycb_07_soup_can_tot_occl.xml"
# SCENE="tests/ycb_08_chips_tot_occl.xml"
# SCENE="tests/ycb_09_mustard_tot_occl.xml"
# SCENE="tests/ycb_clutter.xml"
# SCENE="tests/ycb_fruit.xml"
ARG="$1"
[ -z "$ARG" ] || SCENE=$(python -c "import os; print(os.path.relpath('$ARG','$DIR'))")
GUI=true
[ -z "$2" ] || GUI=false
OUT_PREFIX="OUT"
[ -z "$3" ] || OUT_PREFIX="$3"
PROMPT=35
GT=g
[ -z "$4" ] || PROMPT="$4"
[ -z "$4" ] || GT=r

alacritty -e bash -ic "roslaunch lab_vbnpm control.launch is_sim:=true gui_sim:=$GUI gui_plan:=$GUI scene:=$SCENE" &
# alacritty -e bash -ic "roslaunch --wait lab_vbnpm perception.launch" &
# alacritty -e bash -ic "roslaunch --wait lab_vbnpm perception.launch client_only:=true" &

# alacritty -e bash -ic "roslaunch graspnet_ros container.launch" &
# alacritty -e bash -ic "roslaunch gpg_ros container.launch" &

# CONFIG="$DIR/scripts/grasp_planner/gpd_ros_large_bite.cfg"
# alacritty -e bash -ic "roslaunch --wait gpg_ros container.launch config:=$CONFIG" &
# alacritty -e bash -ic "roslaunch --wait gpd_docker container.launch config:=$CONFIG" &
# alacritty -e bash -ic "roslaunch --wait gpd_docker container2.launch" &
# alacritty -e bash -ic "roslaunch --wait cgn_ros container.launch" &

echo "Press enter once nodes are up!"
read
python $DIR/launch/calibration/sim_params.py
if ! $GUI; then
	$DIR/record_rosbag.sh $OUT_PREFIX.bag &
	echo Scene,$(basename ${SCENE%.xml}) > $OUT_PREFIX.csv
	# python $DIR/tests/curobo_open_loop.py s $GT "$PROMPT" p $OUT_PREFIX.csv
	# python $DIR/tests/curobo_closed_loop.py s $GT "$PROMPT" p $OUT_PREFIX.csv
	python $DIR/tests/curobo_active_grasp.py s $GT "$PROMPT" p $OUT_PREFIX.csv
	echo Done
	rosnode kill /rosbag_record
	sleep 1
	pkill -ef -9 multiprocessing
	docker ps -q
	docker kill $(docker ps -q)
	rosnode kill -a
	killall -wg roslaunch
fi
