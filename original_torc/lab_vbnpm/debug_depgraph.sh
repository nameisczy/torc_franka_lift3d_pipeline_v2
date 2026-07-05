#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$ROOT_DIR/scripts:$PYTHONPATH"
python3 $ROOT_DIR/scripts/tools/debug_depgraph.py "$@"