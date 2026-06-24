#!/bin/bash

mkdir -p Results
S="Results/wild_places_gpu.json"

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Parameters for Multi-Scale ICP have already been set for Wild Places

# Benchmark sequence V-03.

D="Datasets/Processed/wild_places_v_03.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence V-04.

D="Datasets/Processed/wild_places_v_04.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence K-03.

D="Datasets/Processed/wild_places_k_03.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence K-04.

D="Datasets/Processed/wild_places_k_04.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU
