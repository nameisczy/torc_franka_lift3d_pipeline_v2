#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR=$1
PICK_NUM=$2

if [ -z "$EXPERIMENT_DIR" ] || [ -z "$PICK_NUM" ]; then
  echo "Usage: $0 [EXPERIMENT_DIR] [PICK_NUM]"
  echo "  [EXPERIMENT_DIR]: The directory of an experiment."
  echo "  [PICK_NUM]:       The pick number to inspect."
  exit 1
fi

$ROOT_DIR/debug_show.sh dir "$EXPERIMENT_DIR" --files "dep_graph_$PICK_NUM.depgraph" "vlm_dep_graph_$PICK_NUM.depgraph" "img_labeled_$PICK_NUM.png" "dep_graph_grasp_only_$PICK_NUM.depgraph"
