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
