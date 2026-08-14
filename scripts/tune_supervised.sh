#!/bin/bash

S="results/params_supervised_tuned.msgpack"

if [[ -f "$S" ]]; then
	echo "Skipping supervised tuning as '$S' already exists."
	exit 0
fi

export PYTHONPATH=.

uv run python -u experiments/tune_supervised.py
