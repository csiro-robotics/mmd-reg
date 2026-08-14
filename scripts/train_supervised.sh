#!/bin/bash

mkdir -p results
S="results/params_supervised_trained.msgpack"

if [[ -f "$S" ]]; then
	echo "Skipping supervised training as '$S' already exists."
	exit 0
fi

export PYTHONPATH=.

uv run python -u experiments/train_supervised.py
