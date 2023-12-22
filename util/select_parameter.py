# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: generate_graph.py
@Software: Vscode
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from dgl.data.utils import load_graphs
from fns import init_seeds, Newloss
from fns import WeightAndSum, split_trainingData
from fns import weigth_init, collate, ContrastiveLoss
from sklearn.model_selection import StratifiedKFold
import random
import optuna
import time
from dgl.nn.pytorch import GraphConv
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score



class Classifer_SelectSuperparameter(nn.Module):
    def __init__(self,
                 trial
                 ):
        super(Classifer_SelectSuperparameter, self).__init__()

        # superameters
        graph_hidden = trial.suggest_int("graph_hidden", 64, 192, step=64)
        depth = trial.suggest_int("depth", 2,4)
        mlp_layers = trial.suggest_categorical("mlp_layers", [[500, 100], [300, 50],[150,30]])
        dropout = trial.suggest_float("dropout", 0, 0.2, step=0.1)

        # first layer GCN
        self.gcn1 = GraphConv(38, graph_hidden, norm='both', weight=True, bias=True)
        self.gcn1_bn = torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        # the k layer GCN
        self.gcn2 = nn.ModuleList(
            [GraphConv(graph_hidden, graph_hidden, norm='both', weight=True, bias=True) for _ in
             range(depth - 1)])  #
        self.gcn2_bn = nn.ModuleList(
            [torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in
             range(depth - 1)])

        # Apply attention mechanism to each atom.
        self.readout = WeightAndSum(graph_hidden)

        # first FC
        self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
        self.bn1 = torch.nn.BatchNorm1d(mlp_layers[0], eps=1e-05, momentum=0.1,
                                        affine=True, track_running_stats=True)
        # the k layer FC
        self.linears = nn.ModuleList(
            [nn.Linear(mlp_layers[i], mlp_layers[i + 1]) for i in range(len(mlp_layers) - 1)])
        self.bns = nn.ModuleList([torch.nn.BatchNorm1d(mlp_layers[i + 1], eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True) for i in
                                  range(len(mlp_layers) - 1)])

        # last layer
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
    
def SelectSuperParameters(trial):
    net = Classifer_SelectSuperparameter(trial).to(device)
    net.apply(weigth_init)
    learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-4])
    Batch_size = trial.suggest_int("Batch_size", 16, 48, step=16)
    data1 = list(zip(train_x, train_y))
    data2 = list(zip(valid_x, valid_y))
    print("train_data:", len(data1), "valid_data:", len(data2))

    drop_token1 = False
    # bn>1
    if train_y.shape[0] % Batch_size == 1:
        drop_token1 = True
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_dataset = DataLoader(data1, Batch_size, shuffle=True, drop_last=drop_token1,
                               collate_fn=collate)  
    valid_dataset = DataLoader(data2, Batch_size, shuffle=False, drop_last=False, collate_fn=collate)  #



    # contrastive learning 
    epoch = 0
    contras_criterion = ContrastiveLoss()
    while epoch < num_epochs_s:
        train_loss_pretrain_epoch = 0
        for _, (bg, label) in enumerate(train_dataset):
            bg = bg.to(device)
            atom_feats = bg.ndata["h"].to(device)
            atom_feats.requires_grad = True
            label = label.to(device)
            pred = net(bg, atom_feats)[0]
            pred = pred.to(device).float()
            take_index = []
            for i in range(len(torch.where(label != -1)[0])):
                take_index.append((torch.where(label != -1)[0][i] * 67 + torch.where(label != -1)[1][
                    i]).cpu().detach().numpy().tolist())
            take_index = torch.tensor(take_index).to(device)
            label = torch.take(label, take_index).view(len(torch.take(label, take_index)), 1).int()
            pred_contrast = torch.take(pred, take_index).view(len(torch.take(pred, take_index)), 1)
            label_contrast = []
            contrast_len = len(label) // 2
            label1 = label[:contrast_len]
            label2 = label[contrast_len:contrast_len * 2]
            for i in range(contrast_len):
                xor_label = (label1[i] ^ label2[i])
                label_contrast.append(xor_label.unsqueeze(0))
            label_contrast = torch.cat(label_contrast)
            pred_contrast1 = pred_contrast[:contrast_len]
            pred_contrast2 = pred_contrast[contrast_len:contrast_len * 2]
            loss_contrast = contras_criterion(pred_contrast1, pred_contrast2, label_contrast)
            optimizer.zero_grad()
            loss_contrast.backward(retain_graph=True)
            optimizer.step()
            train_loss_pretrain_epoch = train_loss_pretrain_epoch + loss_contrast.detach().cpu()
        epoch += 1


    train_loss_record = []
    valid_loss_record = []
    train_roc_record = []
    valid_roc_record = []
    epoch = 0
    p = []
    for _ in range(train_y.shape[1]):
        yi = train_y[:, _]
        p.append(torch.sum(yi == 0) * 1.0 / (torch.sum(yi == 1) + torch.sum(yi == 0)))
    p = torch.FloatTensor(p).to(device)  
    criterion = Newloss(p=p)  

    while epoch < num_epochs_s:
        y_pred_train_epoch = []
        y_true_train_epoch = []
        y_pred_valid_epoch = []
        y_true_valid_epoch = []

        train_loss_epoch = 0
        valid_loss_epoch = 0
        train_n_epoch = 0
        valid_n_epoch = 0

        start = time.perf_counter()

        net.train()
        for _, (bg, label) in enumerate(train_dataset):
            bg = bg.to(device)
            atom_feats = bg.ndata["h"].to(device)
            atom_feats.requires_grad = True
            label = label.to(device)
            pred = net(bg, atom_feats)[0]
            loss, N = criterion(pred, label)
            loss_mean = torch.mean(loss)
            optimizer.zero_grad()
            loss_mean.backward(retain_graph=True)
            optimizer.step()

            train_loss_epoch = train_loss_epoch + loss.detach().cpu() * N.detach().cpu()
            train_n_epoch = train_n_epoch + N.detach().cpu()
            y_pred_train_epoch.append(pred.detach().cpu())
            y_true_train_epoch.append(label.detach().cpu())

        y_pred_train_epoch = torch.cat(y_pred_train_epoch, dim=0).detach().cpu().numpy()
        y_true_train_epoch = torch.cat(y_true_train_epoch, dim=0).detach().cpu().numpy()
        train_loss_epoch_mean = (torch.sum(train_loss_epoch) / torch.sum(train_n_epoch)).item()  
        train_loss_record.append(train_loss_epoch_mean)
        torch.cuda.empty_cache()  

        train_roc = []
        for _ in range(train_y.shape[1]):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            train_roc.append(roc_auc_score(y_true[y_true != -1], y_pred[y_true != -1]))
        train_roc_record.append(np.mean(train_roc))

        # valid
        net.eval()
        for _, (bg, label) in enumerate(valid_dataset):
            with torch.no_grad():
                bg = bg.to(device)
                atom_feats = bg.ndata["h"].to(device)
                atom_feats.requires_grad = False
                label = label.to(device)
                pred = net(bg, atom_feats)[0]

                loss, N = criterion(pred, label)
                valid_loss_epoch = valid_loss_epoch + loss.detach().cpu() * N.detach().cpu()
                valid_n_epoch = valid_n_epoch + N.detach().cpu()
                y_pred_valid_epoch.append(pred.detach().cpu())
                y_true_valid_epoch.append(label.detach().cpu())

        y_pred_valid_epoch = torch.cat(y_pred_valid_epoch, dim=0).detach().cpu().numpy()
        y_true_valid_epoch = torch.cat(y_true_valid_epoch, dim=0).detach().cpu().numpy()
        valid_loss_epoch_mean = (torch.sum(valid_loss_epoch) / torch.sum(valid_n_epoch)).item()  # 按照样本
        valid_loss_record.append(valid_loss_epoch_mean)
        torch.cuda.empty_cache()

        valid_roc = []
        for _ in range(train_y.shape[1]):
            y_pred = y_pred_valid_epoch[:, _]
            y_true = y_true_valid_epoch[:, _]
            valid_roc.append(roc_auc_score(y_true[y_true != -1], y_pred[y_true != -1]))
        valid_roc_record.append(np.mean(valid_roc))
        torch.cuda.empty_cache()
        epoch = epoch + 1
    return np.mean(valid_roc_record)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_i_ori = '/home/liuhx/shukai/refer/AMGC/dataset/internal_dataset'
    path_e_ori = '/home/liuhx/shukai/refer/AMGC/dataset/external_dataset'

    #load internal dataset
    glist, _ = load_graphs(os.path.join(path_i_ori , 'mul_ori_data.bin'))
    y = np.load(os.path.join(path_i_ori, "y.npy"), allow_pickle=True)
    train_x,train_y,valid_x,valid_y,test_x,test_y= split_trainingData(glist , y ,0)


    num_epochs_s = 30
    weight_decay = 1e-4
    study = optuna.create_study(direction='maximize')
    study.optimize(SelectSuperParameters, n_trials=20)
    trial = study.best_trial
    print("Best hyperparameters: {}".format(trial.params))
