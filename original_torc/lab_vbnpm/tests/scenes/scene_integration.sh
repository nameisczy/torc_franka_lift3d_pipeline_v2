surface=$1
structure=$2

echo "Args: shelf (s) or table (t), structured (s) or unstructured (u)"

scene_parent_dir="/data/local/kc1317/graspclutter6d_mujoco_sim/scenes/selected_scenes/"

# If surface = s, then set it to "shelf"; if surface = t, then set it to "table"
if [ "$surface" == "s" ]; then
    echo "Surface: shelf"
    surface="shelf"
elif [ "$surface" == "t" ]; then
    echo "Surface: tabletop"
    surface="tabletop"
else
    echo "Invalid surface type. Use 's' for shelf or 't' for tabletop."
    exit 1
fi

# If structure = s, then set it to "structured"; if structure = u, then set it to "unstructured"
if [ "$structure" == "s" ]; then
    echo "Structure: structured"
    structure="structured"
elif [ "$structure" == "u" ]; then
    echo "Structure: unstructured"
    structure="unstructured"
else
    echo "Invalid structure type. Use 's' for structured or 'u' for unstructured."
    exit 1
fi

category="${surface}_${structure}"
scene_dir="$scene_parent_dir/${category}/"
# Stabilize all scenes in the directory

lab_path="$(rospack find lab_vbnpm)"
robot_xml="$lab_path/robots/motoman/mjmodel.xml"
out_dir="$lab_path/tests/scenes/${category}/"

status_file="$out_dir/status.txt"
> "$status_file"  # Clear or create the file at the start

# Iterate through files in scene_dir
echo "Dir $scene_dir"
for file in "$scene_dir"/*; do
    if [[ "${file,,}" == *.csv ]]; then
        echo "Skipping CSV: $file"
        continue
    fi
    echo "Processing $file"

    OUTFILE="$out_dir/$(basename "$file")"

    # Call the scene stabilization script
    python "$lab_path/scripts/execution_scene/mjcf_robot_scene_integrator.py" "$robot_xml" "$file" "$OUTFILE"

    if [ -f "$OUTFILE" ]; then
        echo "$(basename "$file"): SUCCESS" >> "$status_file"
    else
        echo "$(basename "$file"): FAILED" >> "$status_file"
    fi
done