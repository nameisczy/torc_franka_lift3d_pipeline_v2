#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$ROOT_DIR/docker/docker_run_ros.sh "$@"