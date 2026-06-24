#!/bin/bash

mkdir -p Results
S="Results/pcpnet_gpu.json"

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Parameters for algorithms, other than MMD-Reg, have already been set for PCPNet

# Benchmark split for gradient sampling density.

D="Datasets/Processed/pcpnet_gradient.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark split for high noise.

D="Datasets/Processed/pcpnet_high_noise.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark split for striped sampling density.

D="Datasets/Processed/pcpnet_striped.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark splits to measure GPU runtime for different numbers of points.

for i in 1 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="Datasets/Processed/pcpnet_time_${I}k.hdf5"
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 
done
