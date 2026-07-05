#!/usr/bin/env bash
for t in $(jq keys valid_scenes.json | head -n-1 | tail -n+2 | tr -d ',"')
do
	if [ $t == "shelf_structured" ]
	then
		continue
	fi
	for s in $(jq .$t  valid_scenes.json | head -n-1 | tail -n+2 | tr -d ',"')
	do
		INFILE="$t/$s"
		echo In File: $t/$s
		OUTFILE="adjusted/${t}_${s:5}"
		python adjust_scene.py $INFILE $OUTFILE
		echo Out File: $OUTFILE
		read -p "Press enter to continue..."
		# sed -i 's#../../##g' $OUTFILE
	done
	
done

