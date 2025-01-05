#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment variables from .env file
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "Error: .env file not found in $SCRIPT_DIR"
    exit 1
fi

exec $CONDA_EXEC run -n $CONDA_ENV python "$SCRIPT_DIR/run_leaderboard.py" --config "$SCRIPT_DIR/$CONFIG_FILE"