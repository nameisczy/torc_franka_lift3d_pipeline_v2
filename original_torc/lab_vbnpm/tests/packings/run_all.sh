#!/usr/bin/env bash
DIR="$(rospack find lab_vbnpm)"
for FILE in packings_small/solution_7*.json
do
	XML=${FILE::-5}.xml
	$DIR/tests/packings/a_ur5e.sh $XML false
	sleep 2
	$DIR/killall.sh
	sleep 2
done
