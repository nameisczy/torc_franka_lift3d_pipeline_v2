FOLDERS=$(grep -zlRP \
    --include='output.csv' \
    'Grasp Success,True(?:.*\n){3}.*Retract Success,False' \
    experiments/runs/ \
    | xargs -I {} dirname {})

while IFS= read -r folder; do
    echo "Inspecting folder: $folder"
    ./mujoco_state_inspector.sh "$folder" -z
done <<< "$FOLDERS"