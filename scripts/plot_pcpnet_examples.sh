#!/bin/bash

mkdir -p results/figures

export PYTHONPATH=.
export JAX_PLATFORMS="cpu"
export JAX_DEFAULT_MATMUL_PRECISION="highest"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1

uv run python -u experiments/plot_pcpnet_examples.py
