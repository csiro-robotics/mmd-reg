#!/bin/bash

mkdir -p Datasets/Processed

export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Ensure the Python virtual environment is active.

# Process datasets for fully-overlapping clean point-clouds

python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_clean_train.hdf5 --split train --same_sample --categories airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp
python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_clean_val.hdf5   --split test  --same_sample --rotate --translate --categories airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp
python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_clean_test.hdf5  --split test  --same_sample --rotate --translate --categories laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox

# Process datasets for partially-overlapping noisy point-clouds

python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_partial_train.hdf5 --split train --categories airplane bathtub bed bench bookshelf car chair curtain desk door dresser glass_box guitar keyboard
python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_partial_val.hdf5   --split test  --crop --jitter --rotate --translate --categories airplane bathtub bed bench bookshelf car chair curtain desk door dresser glass_box guitar keyboard
python -u process_modelnet40.py --save_path Datasets/Processed/modelnet40_partial_test.hdf5  --split test  --crop --jitter --rotate --translate --categories laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table toilet tv_stand wardrobe xbox
