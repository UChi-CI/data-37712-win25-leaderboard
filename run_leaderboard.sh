#!/bin/bash

source ".env"
exec $CONDA_EXEC run -n $CONDA_ENV python run_leaderboard.py --config $CONFIG_FILE