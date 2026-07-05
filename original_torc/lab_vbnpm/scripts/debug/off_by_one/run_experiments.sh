#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath $SCRIPT_DIR/../../..)"
echo $ROOT_DIR
cd $ROOT_DIR

control_c() {
    echo -e "\nCtrl+C detected. Exiting."
    exit 1
}

trap control_c SIGINT

./7kill.sh

# Loop indefinitely until ctrl+C
while true; do
    ./run_experiment.sh trial unstructured --rosbag --server --method dg_only
done