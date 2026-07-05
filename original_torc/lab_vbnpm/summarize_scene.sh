#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 $ROOT_DIR/experiments/summarize_scene.py "$@"