#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 $ROOT_DIR/experiments/run_experiment.py "$@"