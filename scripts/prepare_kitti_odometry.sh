#!/bin/bash

mkdir -p datasets/processed

export JAX_PLATFORMS="cpu"
export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Process sequences 07, 08, 09 and 10.

uv run python -u data_preprocessing/process_kitti_odometry.py --save_path datasets/processed/kitti_odometry_07.hdf5 --sequence 07
uv run python -u data_preprocessing/process_kitti_odometry.py --save_path datasets/processed/kitti_odometry_08.hdf5 --sequence 08
uv run python -u data_preprocessing/process_kitti_odometry.py --save_path datasets/processed/kitti_odometry_09.hdf5 --sequence 09
uv run python -u data_preprocessing/process_kitti_odometry.py --save_path datasets/processed/kitti_odometry_10.hdf5 --sequence 10
