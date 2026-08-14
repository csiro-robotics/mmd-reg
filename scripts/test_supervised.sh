#!/bin/bash

T="results/params_supervised_tuned.msgpack"

if [[ ! -f "$T" ]]; then
	echo "Skipping supervised testing as '$T' does not exist."
	exit 0
fi

export PYTHONPATH=.
export JAX_DEFAULT_MATMUL_PRECISION="bfloat16"

uv run python -u experiments/test_supervised.py
