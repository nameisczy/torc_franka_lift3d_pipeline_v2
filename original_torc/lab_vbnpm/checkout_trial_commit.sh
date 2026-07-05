#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  checkout <trial_dir>  Checkout tests/scenes for the git commit in trial's info_trial.json"
    echo "  reset                 Reset tests/scenes to HEAD of current branch"
    exit 1
fi

COMMAND="$1"

case "$COMMAND" in
    reset)
        echo "Resetting tests/scenes to HEAD..."
        cd "$SCRIPT_DIR"
        git checkout HEAD -- tests/scenes experiments/data/scene_summary.csv 2>&1 || {
            echo "Error: failed to reset tests/scenes to HEAD"
            exit 1
        }
        echo "Successfully reset tests/scenes to HEAD"
        exit 0
        ;;
    checkout)
        if [[ $# -lt 2 ]]; then
            TRIAL_DIR="${SCRIPT_DIR}/experiments/runs"
        else
            TRIAL_DIR="$2"
        fi
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        exit 1
        ;;
esac

INFO_FILE=$(find "$TRIAL_DIR" -name "info_trial.json" -print -quit)

if [[ -z "$INFO_FILE" ]]; then
    echo "Error: info_trial.json not found under $TRIAL_DIR"
    exit 1
fi

# Extract git_commit from JSON
GIT_COMMIT=$(grep -o '"git_commit"[[:space:]]*:[[:space:]]*"[^"]*"' "$INFO_FILE" | sed 's/.*"\([^"]*\)".*/\1/')

if [[ -z "$GIT_COMMIT" ]]; then
    echo "Error: git_commit not found in $INFO_FILE"
    exit 1
fi

echo "Checking out tests/scenes for commit: $GIT_COMMIT"

cd "$SCRIPT_DIR"
git checkout "$GIT_COMMIT" -- tests/scenes experiments/data/scene_summary.csv 2>&1 || {
    echo "Warning: failed to checkout tests/scenes for commit $GIT_COMMIT"
    exit 1
}

echo "Successfully checked out tests/scenes for commit $GIT_COMMIT"
