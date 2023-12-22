# AMGC: a multiple task graph neutral network for epigenetic targets profiling 

![](https://github.com/shukai1997/AMGC/blob/main/Figure1.png)

## Contents

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo--reproduction-ligand-docking-on-pdbbind-core-set)

## Overview 

AMGC (adaptive multi-task graph convolutional network with contrastive learning) is a novel GCN-based multi-task model for predicting the inhibition profiles of small molecules against 67 epigenetic targets.

## Software Requirements

### Python Dependencies

Dependencies for KarmaDock:

```
- python 3.8.12
- DGL 0.6.1 
- PyTorch 1.12.0 
- dgllife 0.2.9 
- RDKIT (recommended version 2021.09.2) 
```

## Installation Guide

### download this repo

```
git clone https://github.com/shukai1997/AMGC.git
```

### install amgc_env

you can install the env via yaml file

```
cd KarmaDock
conda env create -f amgc_env.yaml
```

## Demo & Reproduction: ligand docking on PDBBind core set

Assume that the project is at `/root` and therefore the project path is /root/KarmaDock.

### 1. Download PDBBind dataset

You can download the PDBBind 2020 core set without preprocessing from the [PDBBind website](http://pdbbind.org.cn/index.php)
OR you can download [the version](https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1) where protein files were prepared by Schrodinger. 
```
cd /root/KarmaDock
weget https://zenodo.org/record/7788083/files/pdbbind2020_core_set.zip?download=1
unzip -q pdbbind2020_core_set.zip?download=1
```

### 2. Preprocess PDBBind data

The purpose of this step is to identify residues that are within a 12Å radius of any ligand atom and use them as the pocket of the protein. The pocket file (xxx_pocket_ligH12A.pdb) will also be saved on the `complex_file_dir`.

```
cd /root/KarmaDock/utils 
python -u pre_processing.py --complex_file_dir ~/your/PDBBindDataset/path
```
e.g.,
```
cd /root/KarmaDock/utils 
python -u pre_processing.py --complex_file_dir /root/KarmaDock/pdbbind2020_core_set
```

### 3. Generate graphs based on protein-ligand complexes

This step will generate graphs for protein-ligand complexes and save them (*.dgl) to `graph_file_dir`.

```
cd /root/KarmaDock/utils 
python -u generate_graph.py 
--complex_file_dir ~/your/PDBBindDataset/path 
--graph_file_dir ~/the/directory/for/saving/graph 
--csv_file ~/path/of/csvfile/with/dataset/split
```
e.g.,
```
mkdir /root/KarmaDock/test_graph
cd /root/KarmaDock/utils 
python -u generate_graph.py --complex_file_dir /root/KarmaDock/pdbbind2020_core_set --graph_file_dir /root/KarmaDock/test_graph --csv_file /root/KarmaDock/pdbbind2020.csv
```

### 4. ligand docking

This step will perform ligand docking (predict binding poses and binding strengthes) based on the graphs. (finished in about 0.5 min)

```
cd /root/KarmaDock/utils 
python -u ligand_docking.py 
--graph_file_dir ~/the/directory/for/saving/graph 
--csv_file ~/path/of/csvfile/with/dataset/split 
--model_file ~/path/of/trained/model/parameters 
--out_dir ~/path/for/recording/BindingPoses&DockingScores 
--docking Ture/False  whether generating binding poses
--scoring Ture/False  whether predict binding affinities
--correct Ture/False  whether correct the predicted binding poses
--batch_size 64 
--random_seed 2023 
```
e.g.,
```
mkdir /root/KarmaDock/test_result
cd /root/KarmaDock/utils 
python -u ligand_docking.py --graph_file_dir /root/KarmaDock/test_graph --csv_file /root/KarmaDock/pdbbind2020.csv --model_file /root/KarmaDock/trained_models/karmadock_docking.pkl --out_dir /root/KarmaDock/test_result --docking True --scoring True --correct True --batch_size 64 --random_seed 2023
```
