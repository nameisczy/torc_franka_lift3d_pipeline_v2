#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../.."
cd $WS_DIR
docker build -f $DOCKER_DIR/Dockerfile -t lab_vbnpm .
if [[ "$1" == "test" ]]; then
  docker run -it lab_vbnpm
fi