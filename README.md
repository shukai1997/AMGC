# AMGC
## Description
AMGC (adaptive multi-task graph convolutional network with contrastive learning) is a novel GCN-based multi-task model for predicting the inhibition profiles of small molecules against 67 epigenetic targets, which shows the state-of-art predicted performance.
# Installation
## Known Installation Issues
### The following versions must be used in order to use the pretrained models:
- python 3.8.12
- DGL 0.6.1 [https://www.dgl.ai/pages/start.html]
- PyTorch 1.12.0 [https://pytorch.org/get-started/locally/]
- dgllife 0.2.9 [https://github.com/awslabs/dgl-lifesci]
- RDKIT (recommended version 2021.09.2) [https://github.com/rdkit/rdkit]
# The structure of the code is as follows:
    In Data:
    - Dataset for training, validation and test the model
    - The external test sets
    - smile_standardize.py : Standize the molecule smiles
    In Model:
    - graph_generation.py : conver the molecule smiles to molecule graph
    - model_architecture.py : the architecture of the AMGC model
    - model_train.py : Train the model 
    - hyperpameter.py : the best hyperparameter combination 
    - explain.py : atom weights visualization for given compound
    - mul_task.pth : the well-trained AMGC model
 # Note
  If you want to use the AMGC for epigenetic targets fishing, please visit http://cadd.zju.edu.cn/amgc/

  
    

