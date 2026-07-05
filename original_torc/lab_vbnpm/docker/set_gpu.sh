#!/usr/bin/env bash
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sed -i "s/\['.'\]/\['$1'\]/" $CUR_DIR/docker-compose.yaml
