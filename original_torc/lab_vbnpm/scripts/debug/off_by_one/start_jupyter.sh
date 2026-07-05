#!/bin/bash
NOTEBOOK_PATH=$(dirname "$(realpath "$0")")
ROOT_PATH=$(realpath "$NOTEBOOK_PATH/../../..")
SETUP_PATH="$(realpath "$ROOT_PATH/../../devel")/setup.bash"
SCRIPTS_PATH="$ROOT_PATH/scripts"
source "$SETUP_PATH"
export PYTHONPATH="$SCRIPTS_PATH:$PYTHONPATH"

echo "Starting Jupyter Notebook"

echo "Sourced Catkin setup.bash"
echo "  SETUP_PATH:   $SETUP_PATH"
echo "  SCRIPTS_PATH: $SCRIPTS_PATH"
echo "  PYTHONPATH:   $PYTHONPATH"

jupyter notebook --no-browser --port 8888 --ip='*' --NotebookApp.token='' --NotebookApp.password=''