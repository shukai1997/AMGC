# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: generate_graph.py
@Software: Vscode
"""

import torch
from dgllife.utils import mol_to_bigraph
from dgllife.utils import CanonicalBondFeaturizer
from torch.utils.data import DataLoader
from rdkit import Chem
import dgl
from torch.utils.data import Dataset
import torch
import torch.optim
from torch.utils.data import DataLoader
import dgl
import os
import sys
import argparse
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from dataset.smile_standardize import standardize
from dataset.ligand_feature import MyAtomFeaturizer
from fns import collate,WeightAndSum
from architecture.amgc_architecture import Classifer
import pandas as pd



def generate_graph(smiles):
    """
    Converts SMILES into graph with features.
    Parameters
    smiles: SMILES representation of the moelcule of interest
            type smiles: list
    return: DGL graph with features
            rtype: list

    """
    atom = MyAtomFeaturizer()
    bond = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    graph = []
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        Chem.SanitizeMol(mol)
        g = mol_to_bigraph(mol,
                           node_featurizer=atom,
                           edge_featurizer=bond,
                           add_self_loop=True)
        graph.append(g)
    return graph

class pred_data(Dataset):
    def __init__(self, graph, smiles):
        self.smiles = smiles
        self.graph = graph
        self.lens = len(smiles)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        return self.smiles[item], self.graph[item]

def computer_score(smile):
    smile = standardize(smile)
    
    ls_smi = []
    if isinstance(smile, list):
        ls_smi = smile
    else:
        ls_smi.append(smile)
    graph = generate_graph(ls_smi)
    data = pred_data(graph=graph, smiles=ls_smi)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate)
    dataset = data_loader.dataset
    smiles, graph = dataset[0]
    bg = dgl.batch([graph])
    atoms_feat = bg.ndata['h']
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    bg = bg.to(device)
    atoms_feat = atoms_feat.to(device)
    model = torch.load('/home/liuhx/shukai/refer/AMGC/train_models/mul_task0.pth',map_location=torch.device('cpu'))
    model1 = model['model_state_dict'].eval().to(device)
    score = model1(bg,atoms_feat)[0]
    score = score.cpu().detach().numpy()
    score = score[0,:].tolist()
    score = [round(i,4) for i in score]
    return smile, score

def generate_dataframe(smile, score):
    score = score
    data_columns = pd.read_csv(
        '/home/liuhx/shukai/refer/AMGC/train_models/columns.csv')
    columns = list(data_columns['columns'])
    data1 = pd.read_csv(
        '/home/liuhx/shukai/refer/AMGC/train_models/target_info.csv')
    data1['ChEMBL ID'] = data1['ChEMBL ID'].astype('category')
    data1['ChEMBL ID'].cat.reorder_categories(columns, inplace=True)
    data1.sort_values('ChEMBL ID', inplace=True)
    data2 = data1
    data2['Possibility'] = score
    data2['Status'] = ['Predicted'] * 67
    data2 = data2.sort_values('Possibility', ascending=False)
    for i in range(data2.shape[0]):
        if list(data2['Possibility'])[i] <= 0.5:
            break
    index_num = i
    data2 = data2.iloc[:index_num, :]

    data_all = pd.read_csv(
        '/home/liuhx/shukai/refer/AMGC/train_models/multi_task_mywork.csv')
    flag = False
    for i in range(data_all.shape[0]):
        if data_all.iloc[:, 0][i] == smile:
            flag = True
            break
    index_num1 = i
    if flag:
        true_value = [i if i != 1 else int(i) for i in list(data_all.iloc[index_num1, :])[1:]]
        data3 = data1
        data3['Possibility'] = true_value
        data3['Status'] = ['True'] * 67
        rm = []
        for i in range(67):
            if true_value[i] != 1:
                rm.append(i)
        data3 = data3.drop(rm)
        data4 = data3.append(data2)
        data4 = data4.drop_duplicates(['ChEMBL ID'])
        return data4

    else:
        return data2


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--smile', type=str,
                        default='CC1=C(C(NC2=CN=C(S(=O)(N)=O)C=C2)=O)C3=C(N=CN(CCN4CCCC4)C3=O)O1',
                        help='the smile')
    argparser.add_argument('--pred_result_path', type=str,
                        default='/home/liuhx/shukai/refer/AMGC/out_dir/pred_result/pred_result.csv',
                        help='the predition result path')

    args = argparser.parse_args()


    smile, score = computer_score(args.smile)
    data = generate_dataframe(smile, score)
    data.to_csv(args.pred_result_path,index=False)