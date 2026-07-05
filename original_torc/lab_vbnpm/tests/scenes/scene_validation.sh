#!/usr/bin/env bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <surface: s|t> <structure: s|u>"
  exit 1
fi

surface=$1
structure=$2

# map short flags to words
if [ "$surface" = "s" ]; then
  surface="shelf"
elif [ "$surface" = "t" ]; then
  surface="tabletop"
else
  echo "Invalid surface: use 's' or 't'"
  exit 1
fi

if [ "$structure" = "s" ]; then
  structure="structured"
elif [ "$structure" = "u" ]; then
  structure="unstructured"
else
  echo "Invalid structure: use 's' or 'u'"
  exit 1
fi

category="${surface}_${structure}"
scene_dir="$(rospack find lab_vbnpm)/tests/scenes/${category}/"
out_dir="$(rospack find lab_vbnpm)/tests/scenes/${category}/"
temp_file="$out_dir/temp.txt"

results_file="$out_dir/scene_target_objects.txt"
> "$results_file"

echo "Validating scenes for category: $category"

for file in "$scene_dir"/*; do
    if [[ "${file,,}" != *.xml ]]; then
        echo "Skipping non-XML: $file"
        continue
    fi
    echo "Processing $file"
    # Strip xml path to the scene ID 
    
    filename=${file##*/}        # → "scene123.xml"
    # strip the “.xml” suffix:
    base=${filename%.xml}       # → "scene123"
    # strip the “scene” prefix:
    scene_idx=${base#scene} 

    python $(rospack find lab_vbnpm)/tests/scenes/scene_validation.py "$surface" "$structure" "$scene_idx" "$temp_file"

    # Take output from temp_file and append to results_file
    output=$(cat "$temp_file")
    echo "$output" >> "$results_file"

    # names=$(printf "%s\n" "$output" \
    #     | grep "Candidate object name to success:" \
    #     | sed -E "s/.*\{(.*)\}/\1/" \
    #     | tr ',' '\n' \
    #     | sed -nE "s/ *'([^']+)': *True */\1/p"
    # )
done

rm -f "$temp_file"