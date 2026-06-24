#!/bin/bash

mkdir -p Datasets/Processed

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Process sequences 07, 08, 09 and 10.

python -u process_kitti_odometry.py --save_path Datasets/Processed/kitti_odometry_07.hdf5 --sequence 07
python -u process_kitti_odometry.py --save_path Datasets/Processed/kitti_odometry_08.hdf5 --sequence 08
python -u process_kitti_odometry.py --save_path Datasets/Processed/kitti_odometry_09.hdf5 --sequence 09
python -u process_kitti_odometry.py --save_path Datasets/Processed/kitti_odometry_10.hdf5 --sequence 10
