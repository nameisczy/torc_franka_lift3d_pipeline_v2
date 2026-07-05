#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$ROOT_DIR/scripts:$PYTHONPATH"
python3 $ROOT_DIR/scripts/task_planner/curobo_control.py "$@"