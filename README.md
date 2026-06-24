# MMD-Reg

## Environment Setup

Some experiments run MMD-Reg on CPU, while others run it on GPU. 
For simplicity, we recommend creating **two separate virtual environments**: 
one for **CPU-only** runs and one for **GPU** runs.

> **Note:** Install **CPU-only PyTorch** in both environments.  
> PyTorch is used only for data loading, 
> and a CUDA-enabled PyTorch build can conflict with JAX's CUDA setup.

### For MMD-Reg on CPU-only

Create a new virtual environment (e.g., using `venv` or `conda`) 
that includes **Python** and **pip**. 
Then, install the following packages into this environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flax h5py jax jaxopt matplotlib open3d optax pandas probreg tqdm
```

### For MMD-Reg on NVIDIA GPU (CUDA 12)

Create a new virtual environment (e.g., using `venv` or `conda`) 
that includes **Python** and **pip**. 
Then, install the following packages into this environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U "jax[cuda12]"
pip install flax h5py jaxopt matplotlib open3d optax pandas probreg tqdm
```

## Download and Process PCPNet Dataset

After completing these steps, there should be a `Datasets` directory like:

```text
Datasets/
├── PCPNet/
│   ├── armadillo100k.curv
│   └── ...
├── Processed/
│   ├── pcpnet_gradient.hdf5
│   ├── pcpnet_high_noise.hdf5
│   └── ...
└── ...
```

### Step 1: Create the base directory, download data and unzip it, and clean up
```bash
mkdir -p Datasets/PCPNet
curl -o Datasets/pcpnet.zip https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip
unzip Datasets/pcpnet.zip -d Datasets/PCPNet/
rm Datasets/pcpnet.zip
```

### Step 2: Prepare data with CPU-only Python virtual environment active
```bash
bash prepare_pcpnet.sh
```

## Download and Process Wild Places Dataset

After completing these steps, there should be a `Datasets` directory like:

```text
Datasets/
├── Wild-Places/
│   ├── K-03/
│   │   ├── Clouds/
│   │   │   ├── 1639434737.3923593.bin
│   │   │   └── ...
│   │   ├── Clouds_downsampled/
│   │   └── submap_poses.csv
│   ├── K-04/
│   ├── V-03/
│   └── V-04/
├── Processed/
│   ├── wild_places_k_03.hdf5
│   ├── wild_places_k_04.hdf5
│   ├── wild_places_v_03.hdf5
│   ├── wild_places_v_04.hdf5
│   └── ...
└── ...
```

### Step 1: Create the base directory for Wild Places

```bash
mkdir -p Datasets/Wild-Places
```

### Step 2: Download and place the sequences

From <https://data.csiro.au/collection/csiro:56372>, 
download sequences **K-03**, **K-04**, **V-03**, and **V-04**, 
and place them into `Datasets/Wild-Places/`.
We recommend downloading via the `Download files via S3 Client` 
option using the AWS Command Line Interface (AWS CLI). 
To do this, open the collection's **Files** tab, click **Download**, 
and choose `Download files via S3 Client` to obtain the AWS CLI command.

### Step 3: Prepare data with CPU-only Python virtual environment active
```bash
bash prepare_wild_places.sh
```

## Download and Process KITTI Odometry Dataset

After completing these steps, there should be a `Datasets` directory like:

```text
Datasets/
├── KITTI/
│   ├── odometry/
│   │   ├── dataset/
│   │   │   ├── poses/
│   │   │   │   ├── 00.txt
│   │   │   │   ├── 01.txt
│   │   │   │   └── ...
│   │   │   ├── sequences/
│   │   │   │   ├── 00/
│   │   │   │   │   ├── velodyne/
│   │   │   │   │   │   ├── 000000.bin
│   │   │   │   │   │   ├── 000001.bin
│   │   │   │   │   │   └── ...
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   └── times.txt
│   │   │   │   ├── 01/
│   │   │   │   └── ...
├── Processed/
│   ├── kitti_odometry_07.hdf5
│   ├── kitti_odometry_08.hdf5
│   └── ...
└── ...
```

### Step 1: Create the base directory for KITTI Odometry

```bash
mkdir -p Datasets/KITTI/odometry
```

### Step 2: Download data and unzip it

From <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>,
download `data_odometry_calib.zip`, `data_odometry_poses.zip`, and
`data_odometry_velodyne.zip`, and place them into `Datasets/KITTI/odometry/`.

```bash
unzip Datasets/KITTI/odometry/data_odometry_calib.zip -d Datasets/KITTI/odometry/
unzip Datasets/KITTI/odometry/data_odometry_poses.zip -d Datasets/KITTI/odometry/
unzip Datasets/KITTI/odometry/data_odometry_velodyne.zip -d Datasets/KITTI/odometry/
```

### Step 3: Prepare data with CPU-only Python virtual environment active
```bash
bash prepare_kitti_odometry.sh
```

## Download and Process ModelNet40 Dataset

After completing these steps, there should be a `Datasets` directory like:

```text
Datasets/
├── ModelNet40/
│   ├── airplane/
│   │   ├── test/
│   │   │   ├── airplane_0627.off
│   │   │   └── ...
│   │   └── train/
│   ├── bathtub/
│   └── ...
├── Processed/
│   ├── modelnet40_clean_test.hdf5
│   ├── modelnet40_clean_train.hdf5
│   ├── modelnet40_clean_val.hdf5
│   ├── modelnet40_partial_test.hdf5
│   ├── modelnet40_partial_train.hdf5
│   ├── modelnet40_partial_val.hdf5
│   └── ...
└── ...
```

### Step 1: Download data and unzip it, and clean up
```bash
mkdir -p Datasets
curl -o Datasets/ModelNet40.zip https://modelnet.cs.princeton.edu/ModelNet40.zip
unzip Datasets/ModelNet40.zip -d Datasets/
rm Datasets/ModelNet40.zip
```

### Step 2: Prepare data with CPU-only Python virtual environment active
```bash
bash prepare_modelnet40.sh
```

## Run Benchmarks

After downloading and processing the PCPNet Dataset,
Wild Places and KITTI Odometry, you can now benchmark MMD-Reg
and the other non-learning-based registration methods.
While running the benchmarks, results can be found in `Results`.

### Run CPU PCPNet benchmarks with CPU-only Python virtual environment active

Ensure the **CPU-only** Python virtual environment is active. 
This benchmark could take **days to run**.

```bash
bash benchmark_cpu_pcpnet.sh
```

### Run GPU PCPNet benchmarks with GPU Python virtual environment active

Ensure the **GPU** Python virtual environment is active. 
This benchmark could take **hours to run**.

```bash
bash benchmark_gpu_pcpnet.sh
```

### Run GPU Wild Places benchmarks with GPU Python virtual environment active

Ensure the **GPU** Python virtual environment is active. 
This benchmark could take **hours to run**.

```bash
bash benchmark_gpu_wild_places.sh
```

### Run GPU KITTI Odometry benchmarks with GPU Python virtual environment active

Ensure the **GPU** Python virtual environment is active. 
This benchmark could take **hours to run**.

```bash
bash benchmark_gpu_kitti_odometry.sh
```

## Train and Test Unsupervised Neural MMD-Reg (G/L)

After downloading and processing the ModelNet40 Dataset, you can now 
train and test unsupervised Neural MMD-Reg.
Ensure the **GPU** Python virtual environment is active. 
Training could take **days to run**.
To train the model, use

```bash
python -u train_modelnet40_clean.py --dist gaussian
python -u train_modelnet40_clean.py --dist laplace
```

To test the model, after training, use

```bash
JAX_DEFAULT_MATMUL_PRECISION="highest" python -u test_modelnet40_clean.py --dist gaussian
JAX_DEFAULT_MATMUL_PRECISION="highest" python -u test_modelnet40_clean.py --dist laplace
```

## Train, Tune and Test Supervised Neural MMD-Reg

After downloading and processing the ModelNet40 Dataset, you can now 
train, tune and test supervised Neural MMD-Reg.
Ensure the **GPU** Python virtual environment is active. 
Training could take **days to run**.
To train the model, use

```bash
python -u train_modelnet40_partial.py
```

To tune the model, after training, use

```bash
python -u tune_modelnet40_partial.py
```

To test the model, after tuning, use

```bash
python -u test_modelnet40_partial.py
```
