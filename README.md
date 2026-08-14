# MMD-Reg

[ICML 2026]
**Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy**

Rixon Crane, Fahira Afzal Maken, Nicholas Lawrance, Stanislav Funiak,
Kasra Khosoussi, Ming Xu, Russell Tsuchida

## Abstract

We present MMD-Reg, a novel correspondence-free approach to point-cloud
registration that is differentiable and has linear computational complexity in
the number of points. We model registration as a nonlinear least-squares
problem based on the Maximum Mean Discrepancy, approximated using random
Fourier features. The resulting objective can be solved efficiently with
standard methods such as Levenberg-Marquardt, and the solution is
differentiable via the implicit function theorem. This allows MMD-Reg to be
used as a differentiable optimization layer within end-to-end trainable models,
supporting registration under challenging conditions such as poor initial
alignment and partial overlap. We demonstrate this Neural MMD-Reg formulation
by integrating the layer with a set transformer, training the resulting model
in supervised and unsupervised settings, and comparing its performance against
recent learning-based methods. We also evaluate standalone MMD-Reg, comparing
its accuracy and scalability against widely used non-learning-based
registration methods.

## Environment Setup

This project uses <https://github.com/astral-sh/uv> for environment management.
We assume you are using macOS or Linux.

GPU experiments require Linux with an NVIDIA GPU that supports CUDA 12.

Install the dependencies with:

```bash
uv sync
```

Note that you may need to specify the Python 3.12 executable with `--python`.
For example, on an HPC system, you may need to load a Python 3.12 module and
run `uv sync --python "$(which python)"`.

## Download and Process Datasets

Each experiment depends on a specific dataset. You only need to download and
process the datasets required for the experiments you intend to run. Processed
files are written to `datasets/processed/`.

### PCPNet

Download the data, unzip it, and remove the archive:

```bash
mkdir -p datasets/pcpnet
curl -o datasets/pcpnet.zip https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip
unzip datasets/pcpnet.zip -d datasets/pcpnet/
rm datasets/pcpnet.zip
```

Then process the data:

```bash
bash scripts/process_pcpnet.sh
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

After the processed HDF5 files have been generated, you can optionally delete
the `datasets/pcpnet` directory to save disk space.

### Wild Places

Create the base directory:

```bash
mkdir -p datasets/wild_places
```

Then, from <https://data.csiro.au/collection/csiro:56372>, download the `K-03`,
`K-04`, `V-03`, and `V-04` directories and place them in
`datasets/wild_places/`. We recommend downloading the data using the
**Download files via S3 Client** option with the AWS Command Line Interface
(AWS CLI). To do this, open the collection's **Files** tab, click **Download**,
and choose **Download files via S3 Client** to obtain the AWS CLI command.

Then process the data:

```bash
bash scripts/process_wild_places.sh
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

After the processed HDF5 files have been generated, you can optionally delete
the `datasets/wild_places` directory to save disk space.

### KITTI Odometry

Create the base directory:

```bash
mkdir -p datasets/kitti/odometry
```

Then, from <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>, download
`data_odometry_calib.zip`, `data_odometry_poses.zip`, and
`data_odometry_velodyne.zip`, and place them in `datasets/kitti/odometry/`.

Unzip the files:

```bash
unzip datasets/kitti/odometry/data_odometry_calib.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_poses.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_velodyne.zip -d datasets/kitti/odometry/
```

Then process the data:

```bash
bash scripts/process_kitti_odometry.sh
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

After the processed HDF5 files have been generated, you can optionally delete
the `datasets/kitti` directory to save disk space.

### ModelNet40

Download the data, unzip it, and remove the archive:

```bash
mkdir -p datasets
curl -o datasets/ModelNet40.zip https://modelnet.cs.princeton.edu/ModelNet40.zip
unzip datasets/ModelNet40.zip -d datasets/
mv datasets/ModelNet40 datasets/modelnet40
rm datasets/ModelNet40.zip
```

Then process the data:

```bash
bash scripts/process_modelnet40.sh
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

After the processed HDF5 files have been generated, you can optionally delete
the `datasets/modelnet40` directory to save disk space.

## Experiments

Experiment scripts create the `results` directory if it does not already exist,
then save results there.

A copy of our result files is also available in the v1.0.0 release at
<https://github.com/csiro-robotics/mmd-reg/releases/tag/v1.0.0>.

Note that some dataset preprocessing steps and experiments are
non-deterministic, so results may vary between runs.

### PCPNet Benchmarks

Either run the CPU and GPU benchmarks below after processing the PCPNet data or
download our result files from the v1.0.0 release:

```bash
mkdir -p results
curl -L -o results/pcpnet_cpu.json https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/pcpnet_cpu.json
curl -L -o results/pcpnet_gpu.json https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/pcpnet_gpu.json
```

#### CPU

To run the CPU PCPNet benchmarks (which can take **days**), use:

```bash
bash scripts/benchmark_cpu_pcpnet.sh
```

#### GPU

To run the GPU PCPNet benchmarks (which can take **hours**), use:

```bash
bash scripts/benchmark_gpu_pcpnet.sh
```

#### Plot

After processing the PCPNet data and either running both the CPU and GPU PCPNet
benchmarks or downloading our result files above, plot the results with:

```bash
bash scripts/plot_pcpnet_results.sh
```

The plots are saved to `results/figures/`.

### GPU Wild Places Benchmarks

Either run the GPU benchmarks below (which can take **hours**) after processing
the Wild Places data or download our result file from the v1.0.0
release:

```bash
mkdir -p results
curl -L -o results/wild_places_gpu.json https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/wild_places_gpu.json
```

To run the GPU Wild Places benchmarks, use:

```bash
bash scripts/benchmark_gpu_wild_places.sh
```

### GPU KITTI Odometry Benchmarks

Either run the GPU benchmarks below (which can take **hours**) after processing
the KITTI Odometry data or download our result file from the v1.0.0
release:

```bash
mkdir -p results
curl -L -o results/kitti_odometry_gpu.json https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/kitti_odometry_gpu.json
```

To run the GPU KITTI Odometry benchmarks, use:

```bash
bash scripts/benchmark_gpu_kitti_odometry.sh
```

### Unsupervised Neural MMD-Reg with Gaussian Random Frequencies

Train and test the model using the commands below after processing the
ModelNet40 data.

#### Train

Either run the training script below (which can take **days**) or download the
trained model parameters from the v1.0.0 release:

```bash
mkdir -p results
curl -L -o results/params_unsupervised_gaussian_trained.msgpack https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/params_unsupervised_gaussian_trained.msgpack
```

To train the model, use:

```bash
bash scripts/train_unsupervised_gaussian.sh
```

#### Test

After either training the model or downloading the trained model parameters
above, test the model with:

```bash
bash scripts/test_unsupervised_gaussian.sh
```

### Unsupervised Neural MMD-Reg with Laplace Random Frequencies

Train and test the model using the commands below after processing the
ModelNet40 data.

#### Train

Either run the training script below (which can take **days**) or download the
trained model parameters from the v1.0.0 release:

```bash
mkdir -p results
curl -L -o results/params_unsupervised_laplace_trained.msgpack https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/params_unsupervised_laplace_trained.msgpack
```

To train the model, use:

```bash
bash scripts/train_unsupervised_laplace.sh
```

#### Test

After either training the model or downloading the trained model parameters
above, test the model with:

```bash
bash scripts/test_unsupervised_laplace.sh
```

### Supervised Neural MMD-Reg with Laplace Random Frequencies

Train, tune, and test the model using the commands below after processing the
ModelNet40 data.

#### Train

Either run the training script below (which can take **days**) or download the
trained model parameters from the v1.0.0 release:

```bash
mkdir -p results
curl -L -o results/params_supervised_trained.msgpack https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/params_supervised_trained.msgpack
```

To train the model, use:

```bash
bash scripts/train_supervised.sh
```

#### Tune

After training the model or downloading the trained model parameters above,
either run the tuning script below (which can take **hours**) or download the
tuned model parameters from the v1.0.0 release:

```bash
mkdir -p results
curl -L -o results/params_supervised_tuned.msgpack https://github.com/csiro-robotics/mmd-reg/releases/download/v1.0.0/params_supervised_tuned.msgpack
```

To tune the model, use:

```bash
bash scripts/tune_supervised.sh
```

#### Test

After either tuning the model or downloading the tuned model parameters above,
test the model with:

```bash
bash scripts/test_supervised.sh
```
