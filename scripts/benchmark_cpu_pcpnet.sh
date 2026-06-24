#!/bin/bash

mkdir -p Results
S="Results/pcpnet_cpu.json"

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Parameters for algorithms, other than MMD-Reg, have already been set for PCPNet

# Benchmark split for gradient sampling density.

D="Datasets/Processed/pcpnet_gradient.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark split for high noise.

D="Datasets/Processed/pcpnet_high_noise.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark split for striped sampling density.

D="Datasets/Processed/pcpnet_striped.hdf5"

python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark splits to measure CPU runtime for different numbers of points.

for i in 1 2 4 6 8 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="Datasets/Processed/pcpnet_time_${I}k.hdf5"
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR
done

for i in 1 2 4 6 8 10; do
    I=$(printf "%02d" "$i")
    D="Datasets/Processed/pcpnet_time_${I}k.hdf5"
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm CPD
done

# Benchmark splits for different outlier percentages.

for i in 0 1 2 10 20 30 40 50 60 70 80 90 100; do
    I=$(printf "%03d" "$i")
    D="Datasets/Processed/pcpnet_outliers_${I}.hdf5"
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
    python -u benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR
done
