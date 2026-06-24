#!/bin/bash

mkdir -p Datasets/Processed

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Process sequences K-03, K-04, V-03, and V-04.

python -u process_wild_places.py --save_path Datasets/Processed/wild_places_k_03.hdf5 --sequence K-03
python -u process_wild_places.py --save_path Datasets/Processed/wild_places_k_04.hdf5 --sequence K-04
python -u process_wild_places.py --save_path Datasets/Processed/wild_places_v_03.hdf5 --sequence V-03
python -u process_wild_places.py --save_path Datasets/Processed/wild_places_v_04.hdf5 --sequence V-04
