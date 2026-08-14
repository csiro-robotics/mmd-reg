#!/bin/bash

T="results/params_unsupervised_laplace_trained.msgpack"

if [[ ! -f "$T" ]]; then
	echo "Skipping unsupervised Laplace testing as '$T' does not exist."
	exit 0
fi

export PYTHONPATH=.
export JAX_DEFAULT_MATMUL_PRECISION="highest"

uv run python -u experiments/test_unsupervised.py --dist laplace
