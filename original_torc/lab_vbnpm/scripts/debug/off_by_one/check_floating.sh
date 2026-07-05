#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath $SCRIPT_DIR/../../..)"
cd $ROOT_DIR
python3 -m scripts.debug.off_by_one.check_floating "$@"
