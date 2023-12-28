# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: ligand_feature.py
@Software: Vscode
"""
import os
import sys
import argparse
import torch
from predict import generate_graph, pred_data
from fns import collate
from rdkit import Chem
from rdkit import Chem, DataStructs
from IPython.display import display, SVG
from rdkit import Chem
import matplotlib.cm as cm
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import DataLoader
from rdkit import Chem
import dgl
from torch.utils.data import Dataset
import torch
import torch.optim
from torch.utils.data import DataLoader
import dgl
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from fns import collate,WeightAndSum
from architecture.amgc_architecture import Classifer
import pandas as pd




def explain(smiles):
    ls_smi = []
    if isinstance(smiles, list):
        ls_smi = smiles
    else:
        ls_smi.append(smiles)
    graph = generate_graph(ls_smi)
    data = pred_data(graph=graph, smiles=ls_smi)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate)
    dataset = data_loader.dataset
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model = torch.load('../train_models/mul_task0.pth',map_location=torch.device('cpu'))
    model1 = model['model_state_dict'].eval().to(device)
    smiles, graph = dataset[0]
    bg = dgl.batch([graph])
    atoms_feat = bg.ndata['h']
    atom_weight = model1(bg, atoms_feat)[1]
    atom_weights = atom_weight.cpu().detach().numpy()
    score = model1(bg, atoms_feat)[0]
    score = score.cpu().detach().numpy()

    ls = []
    for i in list(range(0, len(atom_weights))):
        ls.append(abs(atom_weights[i][0]))
    min_value = min(ls)
    max_value = max(ls)
    weights = (ls - min_value) / (max_value - min_value)
    mol = Chem.MolFromSmiles(smiles)
    norm = cm.colors.Normalize(vmin=0, vmax=1.00)
    cmap = cm.get_cmap('OrRd')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    atom_colors = {i: plt_colors.to_rgba(weights[i]) for i in range(bg.number_of_nodes())}
    rdDepictor.Compute2DCoords(mol)
    dr = rdMolDraw2D.MolDraw2DSVG(400, 370)
    do = rdMolDraw2D.MolDrawOptions()
    do.bondLineWidth = 4
    do.fixedBondLength = 30
    do.highlightRadius = 4
    dr.SetFontSize(1)
    dr.drawOptions().addAtomIndices = True
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    dr.DrawMolecule(mol, highlightAtoms=range(bg.number_of_nodes()),
                    highlightBonds=[],
                    highlightAtomColors=atom_colors)
    dr.FinishDrawing()
    svg = dr.GetDrawingText()
    svg = svg.replace('svg:', '')
    return svg 


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--smile', type=str,
                        default='C[C@H](CN[C@@H](C(=O)NC1=NC=C(C=C1)C1=CN(C)N=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C#N',
                        help='the smile')
    argparser.add_argument('--pred_result_path', type=str,
                        default='../out_dir/explainability',
                        help='the predition result path')
    args = argparser.parse_args()
    svg = explain(args.smile)
    with open(os.path.join(args.pred_result_path, 'explain.svg'), 'w') as f:
        f.write(svg)
    display(SVG(svg))


