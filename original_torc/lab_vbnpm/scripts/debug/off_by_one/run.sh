#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath $SCRIPT_DIR/../../..)"
echo $ROOT_DIR
cd $ROOT_DIR
./run_ros.sh ./scripts/debug/off_by_one/control_sim.yaml