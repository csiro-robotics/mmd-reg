#!/bin/bash

mkdir -p results
S="results/params_unsupervised_laplace_trained.msgpack"

if [[ -f "$S" ]]; then
	echo "Skipping unsupervised Laplace training as '$S' already exists."
	exit 0
fi

export PYTHONPATH=.

uv run python -u experiments/train_unsupervised.py --dist laplace
