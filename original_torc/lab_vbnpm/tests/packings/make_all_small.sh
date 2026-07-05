#!/usr/bin/env bash
for FILE in packings_small/*.json
do
	XML=${FILE::-5}.xml
	python make_box_scene.py ../../xmls/ur5e_real_shelf.xml $FILE
	python ../mjcf_robot_scene_integrator.py ../../robots/ur5e/ur5e.xml temp.xml tmp.xml
	mv tmp.xml $XML
	sed -i 's#../../#../../../#' $XML
done
