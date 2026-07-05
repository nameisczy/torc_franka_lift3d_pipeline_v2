#!/bin/bash

# Script to extract back.png and color_1.png images from trial experiments
# and organize them into a structured folder hierarchy

TRIAL_PATH="/home/atlinx/projects/lab_ws/src/lab_vbnpm/experiments/runs/trial_2026-03-02_19-05-51__all__human"
OUTPUT_DIR="images"

# Create the output directory structure
echo "Creating directory structure..."
mkdir -p "$OUTPUT_DIR/structured/back"
mkdir -p "$OUTPUT_DIR/structured/front"
mkdir -p "$OUTPUT_DIR/unstructured/back"
mkdir -p "$OUTPUT_DIR/unstructured/front"

# Process all experiment folders
echo "Processing images..."

find "$TRIAL_PATH" -type d -name "*__human" | while read experiment_dir; do
    # Extract the type (structured/unstructured) and metadata from the folder name
    folder_name=$(basename "$experiment_dir")
    
    # Remove the __human suffix
    folder_name="${folder_name%__human}"
    
    # Determine if structured or unstructured
    if [[ "$folder_name" =~ ^structured_ ]]; then
        type="structured"
        # Remove "structured_" prefix to get the trial info
        trial_info="${folder_name#structured_}"
    elif [[ "$folder_name" =~ ^unstructured_ ]]; then
        type="unstructured"
        # Remove "unstructured_" prefix to get the trial info
        trial_info="${folder_name#unstructured_}"
    else
        continue
    fi
    
    # trial_info should be in format: {trial_id}__obj_{object_id}_{grasp_id}
    # Convert to filename format: {trial_id}__{object_id}_{grasp_id}.png
    # But we need to extract just the object and grasp IDs from the object_id part
    
    # Parse the trial_info string
    # Example: 198__obj_000005_0
    trial_id=$(echo "$trial_info" | cut -d'_' -f1)
    obj_and_grasp=$(echo "$trial_info" | sed 's/.*__obj_//')
    # obj_and_grasp should be like "000005_0"
    
    # Create the target filename
    target_filename="${trial_id}__obj_${obj_and_grasp}.png"
    
    # Copy back.png to the back folder
    if [ -f "$experiment_dir/back.png" ]; then
        cp "$experiment_dir/back.png" "$OUTPUT_DIR/$type/back/$target_filename"
        echo "Copied: $experiment_dir/back.png -> $OUTPUT_DIR/$type/back/$target_filename"
    fi
    
    # Copy color_1.png to the front folder
    if [ -f "$experiment_dir/color_1.png" ]; then
        cp "$experiment_dir/color_1.png" "$OUTPUT_DIR/$type/front/$target_filename"
        echo "Copied: $experiment_dir/color_1.png -> $OUTPUT_DIR/$type/front/$target_filename"
    fi
done

echo "Done! Images have been organized in the $OUTPUT_DIR directory."
echo ""
echo "Directory structure created:"
tree "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f | sort
