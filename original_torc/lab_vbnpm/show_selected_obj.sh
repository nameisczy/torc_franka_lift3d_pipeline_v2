#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 $ROOT_DIR/scripts/tools/show_selected_obj.py "$@"