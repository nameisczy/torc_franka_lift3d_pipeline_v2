#!/usr/bin/env bash
for x in $(echo experiments/runs/grasp_planning-*/trial*)
do
	echo -e "$(ls $x | grep -Pv '(csv|json)' | wc -l)\t$x"
done
