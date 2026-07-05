#!/usr/bin/env bash
NUM="$1"
mv /tmp/recording_0000 recording_$NUM
mkdir -p /tmp/recording_0000/color
mkdir -p /tmp/recording_0000/depth
mkdir -p /tmp/recording_0000/mask
mkdir -p /tmp/recording_0000/poses
