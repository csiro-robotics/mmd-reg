# MMD-Reg

## Download and Process Datasets

### PCPNet

Download the data, unzip it, and clean up:

```bash
mkdir -p datasets/pcpnet
curl -o datasets/pcpnet.zip https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip
unzip datasets/pcpnet.zip -d datasets/pcpnet/
rm datasets/pcpnet.zip
```

Then prepare the data:

```bash
bash scripts/prepare_pcpnet.sh
```

There should now be a `datasets` directory like:

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

### Wild Places

Create the base directory:

```bash
mkdir -p datasets/wild_places
```

Then, from <https://data.csiro.au/collection/csiro:56372>, 
download sequences **K-03**, **K-04**, **V-03**, and **V-04**, 
and place them into `datasets/wild_places/`.
We recommend downloading via the `Download files via S3 Client` 
option using the AWS Command Line Interface (AWS CLI). 
To do this, open the collection's **Files** tab, click **Download**, 
and choose `Download files via S3 Client` to obtain the AWS CLI command.

Then prepare the data:

```bash
bash scripts/prepare_wild_places.sh
```

There should now be a `datasets` directory like:

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

### KITTI Odometry

Create the base directory:

```bash
mkdir -p datasets/kitti/odometry
```

Then, from <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>,
download `data_odometry_calib.zip`, `data_odometry_poses.zip`, and
`data_odometry_velodyne.zip`, and place them into `datasets/kitti/odometry/`.
Then unzip them:

```bash
unzip datasets/kitti/odometry/data_odometry_calib.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_poses.zip -d datasets/kitti/odometry/
unzip datasets/kitti/odometry/data_odometry_velodyne.zip -d datasets/kitti/odometry/
```

Then prepare the data:

```bash
bash scripts/prepare_kitti_odometry.sh
```

There should now be a `datasets` directory like:

```text
datasets/
├── kitti/
│   └── odometry/
│       └── dataset/
│           ├── poses/
│           │   ├── 00.txt
│           │   ├── 01.txt
│           │   └── ...
│           └── sequences/
│               ├── 00/
│               │   ├── velodyne/
│               │   │   ├── 000000.bin
│               │   │   ├── 000001.bin
│               │   │   └── ...
│               │   ├── calib.txt
│               │   └── times.txt
│               ├── 01/
│               └── ...
├── processed/
│   ├── kitti_odometry_07.hdf5
│   ├── kitti_odometry_08.hdf5
│   ├── kitti_odometry_09.hdf5
│   ├── kitti_odometry_10.hdf5
│   └── ...
└── ...
```

### ModelNet40

Download the data, unzip it, and clean up:

```bash
mkdir -p datasets
curl -o datasets/ModelNet40.zip https://modelnet.cs.princeton.edu/ModelNet40.zip
unzip datasets/ModelNet40.zip -d datasets/
mv datasets/ModelNet40 datasets/modelnet40
rm datasets/ModelNet40.zip
```

Then prepare the data:

```bash
bash scripts/prepare_modelnet40.sh
```

There should now be a `datasets` directory like:

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
