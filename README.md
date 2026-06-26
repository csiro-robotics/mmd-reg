# MMD-Reg

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv)
for environment management. We assume you are using macOS or Linux.

GPU experiments require Linux with an NVIDIA GPU that supports CUDA 12.

Install the dependencies with:

```bash
uv sync
```

Note: you may need to specify the Python executable with `--python`.
For example, on an HPC system, you may need to load a Python 3.12 module
and run `uv sync --python "$(which python)"`.

## Download and Process Datasets

### PCPNet

Download the data, unzip it, and remove the archive:

```bash
mkdir -p datasets/pcpnet
curl -L -o datasets/pcpnet.zip https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip
unzip datasets/pcpnet.zip -d datasets/pcpnet/
rm datasets/pcpnet.zip
```

Then process the data:

```bash
bash scripts/prepare_pcpnet.sh
```

You should now have a `datasets` directory structured like this:

```text
datasets/
├── pcpnet/
│   ├── armadillo100k.curv
│   └── ...
├── processed/
│   ├── pcpnet_gradient.hdf5
│   ├── pcpnet_high_noise.hdf5
│   └── ...
└── ...
```

After the processed HDF5 files have been generated, you can optionally
delete the `datasets/pcpnet` directory to save disk space.

### Wild Places

Create the base directory:

```bash
mkdir -p datasets/wild_places
```

Then, from <https://data.csiro.au/collection/csiro:56372>, 
download the `K-03`, `K-04`, `V-03`, and `V-04` directories
and place them in `datasets/wild_places/`.
We recommend downloading the data using the **Download files via S3 Client**
option with the AWS Command Line Interface (AWS CLI). 
To do this, open the collection's **Files** tab, click **Download**, 
and choose **Download files via S3 Client** to obtain the AWS CLI command.

Then process the data:

```bash
bash scripts/prepare_wild_places.sh
```

You should now have a `datasets` directory structured like this:

```text
datasets/
├── wild_places/
│   ├── K-03/
│   │   ├── Clouds/
│   │   │   ├── 1639434737.3923593.bin
│   │   │   └── ...
│   │   ├── Clouds_downsampled/
│   │   └── submap_poses.csv
│   ├── K-04/
│   ├── V-03/
│   └── V-04/
├── processed/
│   ├── wild_places_k_03.hdf5
│   ├── wild_places_k_04.hdf5
│   ├── wild_places_v_03.hdf5
│   ├── wild_places_v_04.hdf5
│   └── ...
└── ...
```

After the processed HDF5 files have been generated, you can optionally
delete the `datasets/wild_places` directory to save disk space.

### KITTI Odometry

Create the base directory:

```bash
mkdir -p datasets/kitti/odometry
```

Then, from <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>,
download `data_odometry_calib.zip`, `data_odometry_poses.zip`, and
`data_odometry_velodyne.zip`, and place them in `datasets/kitti/odometry/`.

Unzip the files:

```bash
unzip datasets/kitti/odometry/data_odometry_calib.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_poses.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_velodyne.zip -d datasets/kitti/odometry/
```

Then process the data:

```bash
bash scripts/prepare_kitti_odometry.sh
```

You should now have a `datasets` directory structured like this:

```text
datasets/
├── kitti/
│   └── odometry/
│       ├── dataset/
│       │   ├── poses/
│       │   │   ├── 00.txt
│       │   │   ├── 01.txt
│       │   │   └── ...
│       │   └── sequences/
│       │       ├── 00/
│       │       │   ├── velodyne/
│       │       │   │   ├── 000000.bin
│       │       │   │   ├── 000001.bin
│       │       │   │   └── ...
│       │       │   ├── calib.txt
│       │       │   └── times.txt
│       │       ├── 01/
│       │       └── ...
│       ├── data_odometry_calib.zip
│       ├── data_odometry_poses.zip
│       └── data_odometry_velodyne.zip
├── processed/
│   ├── kitti_odometry_07.hdf5
│   ├── kitti_odometry_08.hdf5
│   ├── kitti_odometry_09.hdf5
│   ├── kitti_odometry_10.hdf5
│   └── ...
└── ...
```

After the processed HDF5 files have been generated, you can optionally
delete the `datasets/kitti` directory to save disk space.

### ModelNet40

Download the data, unzip it, and remove the archive:

```bash
mkdir -p datasets
curl -L -o datasets/ModelNet40.zip https://modelnet.cs.princeton.edu/ModelNet40.zip
unzip datasets/ModelNet40.zip -d datasets/
mv datasets/ModelNet40 datasets/modelnet40
rm datasets/ModelNet40.zip
```

Then process the data:

```bash
bash scripts/prepare_modelnet40.sh
```

You should now have a `datasets` directory structured like this:

```text
datasets/
├── modelnet40/
│   ├── airplane/
│   │   ├── test/
│   │   │   ├── airplane_0627.off
│   │   │   └── ...
│   │   └── train/
│   ├── bathtub/
│   └── ...
├── processed/
│   ├── modelnet40_clean_test.hdf5
│   ├── modelnet40_clean_train.hdf5
│   ├── modelnet40_clean_val.hdf5
│   ├── modelnet40_partial_test.hdf5
│   ├── modelnet40_partial_train.hdf5
│   ├── modelnet40_partial_val.hdf5
│   └── ...
└── ...
```

After the processed HDF5 files have been generated, you can optionally
delete the `datasets/modelnet40` directory to save disk space.

## Experiments

Experiment results are saved to the `results` directory.

### CPU PCPNet Benchmarks

This can take **days to run**. After processing the PCPNet data, use:

```bash
bash scripts/benchmark_cpu_pcpnet.sh
```

### GPU PCPNet Benchmarks

This can take **hours to run**. After processing the PCPNet data, use:

```bash
bash scripts/benchmark_gpu_pcpnet.sh
```

### GPU Wild Places Benchmarks

This can take **hours to run**. After processing the Wild Places data, use:

```bash
bash scripts/benchmark_gpu_wild_places.sh
```

### GPU KITTI Odometry Benchmarks

This can take **hours to run**. After processing the KITTI Odometry data, use:

```bash
bash scripts/benchmark_gpu_kitti_odometry.sh
```

### Train and Test Unsupervised Neural MMD-Reg (G/L)

This can take **days to run**. After processing the ModelNet40 data,
train the models with:

```bash
uv run python -u experiments/train_modelnet40_clean.py --dist gaussian
uv run python -u experiments/train_modelnet40_clean.py --dist laplace
```

After training, test the models with:

```bash
JAX_DEFAULT_MATMUL_PRECISION="highest" uv run python -u experiments/test_modelnet40_clean.py --dist gaussian
JAX_DEFAULT_MATMUL_PRECISION="highest" uv run python -u experiments/test_modelnet40_clean.py --dist laplace
```

### Train, Tune, and Test Supervised Neural MMD-Reg

This can take **days to run**. After processing the ModelNet40 data,
train the model with:

```bash
uv run python -u experiments/train_modelnet40_partial.py
```

After training, tune the model with:

```bash
uv run python -u experiments/tune_modelnet40_partial.py
```

After tuning, test the model with:

```bash
uv run python -u experiments/test_modelnet40_partial.py
```
