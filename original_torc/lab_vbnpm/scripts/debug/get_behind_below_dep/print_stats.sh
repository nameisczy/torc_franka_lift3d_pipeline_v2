#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath $SCRIPT_DIR/../../..)"
python3 $ROOT_DIR/scripts/debug/get_behind_below_dep/print_stats.py "$@"