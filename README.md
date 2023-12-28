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

Dependencies for AMGC:

```
- python 3.8.12
- DGL 0.6.1 
- PyTorch 1.12.0 
- dgllife 0.2.9 
- RDKIT (recommended version 2020.09.5) 
```

## Installation Guide

### download this repo

```
git clone https://github.com/shukai1997/AMGC.git
```

### install amgc_env

you can install the env via yaml file

```
cd AMGC
conda env create -f amgc_envs.yaml
conda activate amgc_envs
```

## Demo & Reproduction: 

Assume that the project is at `/root` and therefore the project path is /root/AMGC. **Please manually adjust the root path if you want to run the code.**

### 1. generate graph files

The purpose of this step is to save graph files for all molecules in the internal and external datasets, and save their labels as npy files.

```
cd /root/AMGC/util
python -u generate_graph.py
```

### 2. Predict epigenetic inhibition profiles from the smiles

This step will generate a csv file which record the prediction result of epigenetic inhibition profiles.

```
cd /root/AMGC/util
python -u predict.py 
--smile ~/your/interested/smile 
--pred_result_path ~/the/directory/for/saving/prediction/result
```
e.g.,
```
mkdir /root/AMGC/test_graph
cd /root/AMGC/utils 
python -u predict.py --smile 'CC1=C(C(NC2=CN=C(S(=O)(N)=O)C=C2)=O)C3=C(N=CN(CCN4CCCC4)C3=O)O1' --pred_result_path '/root/AMGC/out_dir/pred_result/test.csv'
```

### 3. explainability

The purpose of this step to rank the contribution of different atom pairs in the molecule to the final prediction result.

```
cd /root/AMGC/util 
python -u explain.py 
--smile ~/your/interested/smile 
--pred_result_path ~/the/directory/for/saving/explainability/result 
```
e.g.,
```
mkdir /root/AMGC/test_result
cd /root/AMGC/util
python -u explain.py --smile 'C[C@H](CN[C@@H](C(=O)NC1=NC=C(C=C1)C1=CN(C)N=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C#N' --pred_result_path '/root/AMGC/out_dir/explainability'
```
