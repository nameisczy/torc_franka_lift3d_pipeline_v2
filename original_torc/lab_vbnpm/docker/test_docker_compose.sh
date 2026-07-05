#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker compose $@ -f $DOCKER_DIR/docker-compose.yaml up