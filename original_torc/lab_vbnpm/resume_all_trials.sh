#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $ROOT_DIR

# Define the default experiments folder
DEFAULT_FOLDER="./experiments/runs"

# Check if a directory is provided as an argument
# If not, use the default folder
if [ -z "$1" ]; then
  EXPERIMENTS_FOLDER="$DEFAULT_FOLDER"
  echo "No directory provided. Using default: $EXPERIMENTS_FOLDER"
else
  EXPERIMENTS_FOLDER="$1"
fi

# Check if the directory exists
if [ ! -d "$EXPERIMENTS_FOLDER" ]; then
  echo "Error: Directory '$EXPERIMENTS_FOLDER' not found."
  exit 1
fi

# Iterate through each subdirectory within the specified folder
for TRIAL_DIR in "$EXPERIMENTS_FOLDER"/*/; do
  # Check if it's a directory
  if [ -d "$TRIAL_DIR" ]; then
    # Call the resume_trial script for the current subfolder
    echo "Resuming trial in: $TRIAL_DIR"
    ./run_experiment.sh resume_trial "$TRIAL_DIR"
  fi
done

echo "All trials have been processed."