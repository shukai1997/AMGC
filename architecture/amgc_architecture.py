import torch
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from dgl.data.utils import load_graphs
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from functools import partial
from dgl.readout import sum_nodes


class Classifer(nn.Module):
    def __init__(self,
                 graph_hidden,
                 depth,
                 mlp_layers,
                 dropout,
                 ):
        super(Classifer, self).__init__()

        ##第一层GIN
        self.gcn1 = GraphConv(38, graph_hidden, norm='both', weight=True, bias=True)
        self.gcn1_bn = torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        ##第k层GIN
        self.gcn2 = nn.ModuleList(
            [GraphConv(graph_hidden, graph_hidden, norm='both', weight=True, bias=True) for _ in
             range(depth - 1)])  #
        self.gcn2_bn = nn.ModuleList(
            [torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in
             range(depth - 1)])
        #给每个原子加注意力机制
        self.readout = WeightAndSum(graph_hidden)
        # 第一层FC
        self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
        self.bn1 = torch.nn.BatchNorm1d(mlp_layers[0], eps=1e-05, momentum=0.1,
                                        affine=True, track_running_stats=True)
        # 第k层FC
        self.linears = nn.ModuleList(
            [nn.Linear(mlp_layers[i], mlp_layers[i + 1]) for i in range(len(mlp_layers) - 1)])
        self.bns = nn.ModuleList([torch.nn.BatchNorm1d(mlp_layers[i + 1], eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True) for i in
                                  range(len(mlp_layers) - 1)])

        # 最后一层
        self.fc2 = nn.Linear(mlp_layers[-1], 67)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h):

        h = self.gcn1(g, h)
        h = self.gcn1_bn(h)
        h = self.act(self.dropout(h))

        for gcn2, bn2 in zip(self.gcn2, self.gcn2_bn):
            h = gcn2(g, h)
            h = bn2(h)
            h = self.act(self.dropout(h))

        # readout
        g.ndata['h'] = h
        x , atom_weights = self.readout(g, h, get_node_weight=True)

        x = self.bn1(self.fc1(x))
        x = self.dropout(x)
        x = self.act(x)
        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = self.dropout(x)
            x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x , atom_weights

class Classifer_SelectSuperparameter(nn.Module):
    def __init__(self,
                 trial
                 ):
        super(Classifer_SelectSuperparameter, self).__init__()

        ##需要调节的参数
        graph_hidden = trial.suggest_int("graph_hidden", 64, 192, step=64)
        depth = trial.suggest_int("depth", 2,4)
        mlp_layers = trial.suggest_categorical("mlp_layers", [[500, 100], [300, 50],[150,30]])
        dropout = trial.suggest_float("dropout", 0, 0.2, step=0.1)

        ##第一层GIN
        self.gcn1 = GraphConv(38, graph_hidden, norm='both', weight=True, bias=True)
        self.gcn1_bn = torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        ##第k层GIN
        self.gcn2 = nn.ModuleList(
            [GraphConv(graph_hidden, graph_hidden, norm='both', weight=True, bias=True) for _ in
             range(depth - 1)])  #
        self.gcn2_bn = nn.ModuleList(
            [torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in
             range(depth - 1)])

        #给每个原子加注意力机制
        self.readout = WeightAndSum(graph_hidden)

        # 第一层FC
        self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
        self.bn1 = torch.nn.BatchNorm1d(mlp_layers[0], eps=1e-05, momentum=0.1,
                                        affine=True, track_running_stats=True)
        # 第k层FC
        self.linears = nn.ModuleList(
            [nn.Linear(mlp_layers[i], mlp_layers[i + 1]) for i in range(len(mlp_layers) - 1)])
        self.bns = nn.ModuleList([torch.nn.BatchNorm1d(mlp_layers[i + 1], eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True) for i in
                                  range(len(mlp_layers) - 1)])

        # 最后一层
        self.fc2 = nn.Linear(mlp_layers[-1], 67)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h):

        h = self.gcn1(g, h)
        h = self.gcn1_bn(h)
        h = self.act(self.dropout(h))

        for gcn2, bn2 in zip(self.gcn2, self.gcn2_bn):
            h = gcn2(g, h)
            h = bn2(h)
            h = self.act(self.dropout(h))

        # readout
        g.ndata['h'] = h
        x , atom_weights = self.readout(g, h, get_node_weight=True)

        x = self.bn1(self.fc1(x))
        x = self.dropout(x)
        x = self.act(x)
        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = self.dropout(x)
            x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x , atom_weights

class WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )

    def forward(self, g, feats , get_node_weight=True):
        with g.local_scope():
            g.ndata['h'] = feats
            atom_weights = self.atom_weighting(g.ndata['h'])
            g.ndata['w'] = torch.nn.Sigmoid()(self.atom_weighting(g.ndata['h']))
            h_g_sum = sum_nodes(g, 'h', 'w')
        if get_node_weight:
            return h_g_sum, atom_weights
        else:
            return h_g_sum

