
# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: generate_graph.py
@Software: Vscode
"""

import os
import sys
import argparse
import  numpy as np
import pandas as pd
import torch
from dgllife.utils import CanonicalBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem, DataStructs
from dgl.data.utils import save_graphs
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from dataset.ligand_feature import MyAtomFeaturizer




if __name__ == "__main__":
    # change the file path when you run it  
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--internal_dataset', type=str,
                        default='../dataset/internal_dataset/multi_task.csv',
                        help='the complex file path')
    argparser.add_argument('--external_dataset', type=str,
                        default='../dataset/external_dataset/multi_task.csv',
                        help='the graph files path')
    argparser.add_argument('--out_graph_dir', type=str,
                        default='../dataset/internal_dataset',
                        help='the csv file with dataset split')
    args = argparser.parse_args()

    internal_dataset_csv_file = args.internal_dataset
    df = pd.read_csv(internal_dataset_csv_file)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='std_smiles')
    atom_featurizer = MyAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat',self_loop=True)
    graphs = [mol_to_bigraph(m, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer,add_self_loop=True) for m in df.ROMol] #relativelt time-consuming
    graph_labels = {"glabel": torch.tensor(df.index)}
    save_graphs(os.path.join(args.out_graph_dir, "mul_ori_data.bin"), graphs, graph_labels)
    y = df.iloc[:,1:-1]
    y[y!=y]=-1
    y = y.astype("int32")
    np.save(os.path.join(args.out_graph_dir, "y.npy"),y)
