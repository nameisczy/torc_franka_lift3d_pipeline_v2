#!/usr/bin/env bash
ARG="$1"

alacritty -e bash -ic "roslaunch lab_vbnpm ur5e.launch is_sim:=false gui_sim:=true gui_plan:=true" &
echo Calibrate box placements: python ur5e_goto_obj.py $ARG ${ARG::-4}.json
echo Run experiment: python ur5e_move_boxes.py $ARG ${ARG::-4}.json
