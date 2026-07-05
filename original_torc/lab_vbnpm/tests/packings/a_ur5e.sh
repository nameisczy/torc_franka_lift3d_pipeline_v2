#!/usr/bin/env bash
ARG="$1"
DIR="$(rospack find lab_vbnpm)"

SCENE=$(python -c "import os; print(os.path.relpath('$ARG','$DIR'))")
[ -z "$2" ] || GUI="true" && GUI="$2"

alacritty -e bash -ic "roslaunch lab_vbnpm ur5e.launch is_sim:=true gui_sim:=$GUI gui_plan:=$GUI scene:=$SCENE" &
sleep 1
python ur5e_move_boxes.py $ARG ${ARG::-4}.json sim
# python ur5e_goto_obj.py $ARG ${ARG::-4}.json sim
