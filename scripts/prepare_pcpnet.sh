#!/bin/bash

mkdir -p Datasets/Processed

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Process test splits for high noise, gradient sampling density, and striped sampling density.

python -u process_pcpnet.py --save_path Datasets/Processed/pcpnet_gradient.hdf5   --split test_var_density_gradient --number_of_passes 200
python -u process_pcpnet.py --save_path Datasets/Processed/pcpnet_high_noise.hdf5 --split test_high_noise           --number_of_passes 200
python -u process_pcpnet.py --save_path Datasets/Processed/pcpnet_striped.hdf5    --split test_var_density_striped  --number_of_passes 200

# Process splits to measure runtime for different numbers of points.

for i in 1 2 4 6 8 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="Datasets/Processed/pcpnet_time_${I}k.hdf5"
    P=$((i * 1000))
    python -u process_pcpnet.py --save_path "$D" --split test_all --number_of_passes 30 --number_of_points "$P"
done

# Process splits for different outlier percentages.

for i in 0 1 2 10 20 30 40 50 60 70 80 90 100; do
    I=$(printf "%03d" "$i")
    D="Datasets/Processed/pcpnet_outliers_${I}.hdf5"
    O=$((i * 500))
    python -u process_pcpnet.py --save_path "$D" --split test_no_noise --number_of_passes 10 --number_of_outliers "$O"
done
