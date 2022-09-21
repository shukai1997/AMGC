# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: multi_task.py
@Software: PyCharm
"""

import torch
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dgl.data.utils import load_graphs
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_auc_score, precision_recall_curve , auc
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, accuracy_score, \
    balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import time
import random
import optuna
import datetime
from functools import partial
from dgl.readout import sum_nodes
from model_architecture import Classifer, Classifer_SelectSuperparameter, WeightAndSum

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

### calculate the loss for the classification task
class Newloss(nn.Module):
    def __init__(self, p):
        super(Newloss, self).__init__()
        self.p = p

    def forward(self, pred, label):
        pred = pred.to(device).float()
        label = label.to(device).float()

        p = self.p
        b = torch.ones_like(label).to(device)
        b[label == -1] = 0

        P = b.clone().detach().to(device).float()
        P[torch.where(label == 1)] = (p.repeat(label.size(0), 1)[torch.where(label == 1)]).to(device).float()
        P[torch.where(label == 0)] = 1 - (p.repeat(label.size(0), 1)[torch.where(label == 0)]).to(device).float()
        criterion2 = nn.BCELoss(weight=P, reduction='none')

        N = torch.sum(b, dim=0)
        loss = torch.sum(criterion2(pred, label), dim=0) / (N + 1e-16)

        return loss, N

def D(p, z, version='simplified'):
    if version == 'simplified':
        return 1 - torch.pow(torch.abs(p - z), 2)
    else:
        raise Exception

### calculate the loss for the contrastive learning
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_distance = D(output1, output2)

        loss_contrastive = torch.mean((1 - label) * (1 - cos_distance) +
                                      label * cos_distance)

        return loss_contrastive

### select the superparameter-coombination on the valid set.
def SelectSuperParameters(trial):
    net = Classifer_SelectSuperparameter(trial).to(device)
    net.apply(weigth_init)
    learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-4, 1e-3, 5e-3,5e-5,1e-5])
    Batch_size = trial.suggest_int("Batch_size", 16, 64, step=16)
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

        print(
            f'epoch/num_epochs:{epoch + 1}/{num_epochs_s} ' +
            f'train_loss_pretrain_epoch: {train_loss_pretrain_epoch} ' + "\n"
        )

        epoch += 1

    train_loss_record = []
    valid_loss_record = []
    train_roc_record = []
    valid_roc_record = []
    epoch = 0

    p = []
    for _ in range(y1.shape[1]):
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
        for _ in range(y1.shape[1]):
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
        for _ in range(y1.shape[1]):
            y_pred = y_pred_valid_epoch[:, _]
            y_true = y_true_valid_epoch[:, _]
            valid_roc.append(roc_auc_score(y_true[y_true != -1], y_pred[y_true != -1]))
        valid_roc_record.append(np.mean(valid_roc))
        torch.cuda.empty_cache()
        epoch = epoch + 1
    return np.mean(valid_roc_record)

def collate(sample):
    graphs, labels = zip(*sample)
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    return batched_graph, labels

def weigth_init(m):
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

def metric(data ,data1 , random_seed ,  y_pred_train_epoch, y_true_train_epoch):
    if y_pred_train_epoch.shape == y_true_train_epoch.shape:
        mcc = []
        f1 = []
        ba = []
      
        for _ in range(67):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred])
            y_true = y_true[y_true != -1]
            mcc.append(matthews_corrcoef(y_true, y_pred))
            f1.append(f1_score(y_true, y_pred))
            ba.append(balanced_accuracy_score(y_true, y_pred))


        data1['mcc'] = mcc
        data1['f1'] = f1
        data1['ba'] = ba

        mcc_mean = np.mean(mcc)
        f1_mean = np.mean(f1)
        ba_mean = np.mean(ba)

        roc = []
        prc = []
        for _ in range(67):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_true = y_true[y_true != -1]
            roc.append(roc_auc_score(y_true, y_pred))
            prc1, prc2, _ = precision_recall_curve(y_true, y_pred)
            prc.append(auc(prc2, prc1))

        data1['roc'] = roc
        data1['prc'] = prc
        roc_mean = np.mean(roc)
        prc_mean = np.mean(prc)

        print(
            f'The metrci ba is {ba_mean} on the training ' + "\n" +
            f'The metrci f1 is {f1_mean} on the training ' + "\n" +
            f'The metrci mcc is {mcc_mean} on the training ' + "\n" +
            f'The metrci roc is {roc_mean} on the training ' + "\n" +
         
            f'The metrci prc is {prc_mean} on the training ' + "\n"   
        )
        data['seed%s' % random_seed] = ['seed%s' % random_seed, mcc_mean, f1_mean, ba_mean, roc_mean, prc_mean]
        data1 = data1.T
        ### please specify a suitable place for the generated file.  
        data1.to_csv('./test/mul_task_all_%s.csv'%random_seed)
    else:
        y_pred_train_epoch = y_pred_train_epoch[:,:36]
        mcc = []
        f1 = []
        ba = []

        for _ in range(36):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred])
            y_true = y_true[y_true != -1]
            mcc.append(matthews_corrcoef(y_true, y_pred))
            f1.append(f1_score(y_true, y_pred))
            ba.append(balanced_accuracy_score(y_true, y_pred))
 

        data1['mcc'] = mcc
        data1['f1'] = f1
        data1['ba'] = ba

        mcc_mean = np.mean(mcc)
        f1_mean = np.mean(f1)
        ba_mean = np.mean(ba)


        roc = []
        prc = []
        for _ in range(36):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_true = y_true[y_true != -1]
            roc.append(roc_auc_score(y_true, y_pred))
            prc1, prc2, _ = precision_recall_curve(y_true, y_pred)
            prc.append(auc(prc2, prc1))

        data1['roc'] = roc
        data1['prc'] = prc
        roc_mean = np.mean(roc)
        prc_mean = np.mean(prc)

        print(
            f'The metrci ba is {ba_mean} on the training ' + "\n" +
            f'The metrci f1 is {f1_mean} on the training ' + "\n" +
            f'The metrci mcc is {mcc_mean} on the training ' + "\n" +
            f'The metrci roc is {roc_mean} on the training ' + "\n" +
            f'The metrci prc is {prc_mean} on the training ' + "\n" 
        )
        data['seed%s' % random_seed] = ['seed%s' % random_seed, mcc_mean, f1_mean, ba_mean, roc_mean, prc_mean]
        data1 = data1.T
        ### please specify a suitable place for the generated file. 
        data1.to_csv('./etest/mul_task_all_%s.csv' % random_seed)

class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = './model_save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model}, self.filename) 

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def test(net, glist3,  y3,  Batch_size):
    data3 = list(zip(glist3, y3))
    print("test_data:", len(data3))
    drop_token1 = False
    # bn>1
    if y1.shape[0] % Batch_size == 1:
        drop_token1 = True

    test_dataset = DataLoader(data3, Batch_size, shuffle=True, drop_last=drop_token1,collate_fn=collate)

    y_pred_test_epoch = []
    y_true_test_epoch = []
    net.eval()
    for _, (bg, label) in enumerate(test_dataset):
        with torch.no_grad():
            bg = bg.to(device)
            atom_feats = bg.ndata["h"].to(device)
            atom_feats.requires_grad = False
            label = label.to(device)
            pred = net(bg, atom_feats)[0]
            y_pred_test_epoch.append(pred.detach().cpu())
            y_true_test_epoch.append(label.detach().cpu())
    y_pred_test_epoch = torch.cat(y_pred_test_epoch, dim=0).detach().cpu().numpy()
    y_true_test_epoch = torch.cat(y_true_test_epoch, dim=0).detach().cpu().numpy()
    if y_pred_test_epoch.shape == y_true_test_epoch.shape:
        metric(data_test, data_test1, random_seed, y_pred_test_epoch, y_true_test_epoch)
    else:
        metric(data_etest, data_etest1, random_seed, y_pred_test_epoch, y_true_test_epoch)

def train(net, glist1, glist2, y1, y2, learning_rate, weight_decay, Batch_size):
    data1 = list(zip(glist1, y1))
    data2 = list(zip(glist2, y2))
    print("train_data:", len(data1), "valid_data:", len(data2))

    drop_token1 = False
    # bn>1
    if y1.shape[0] % Batch_size == 1:
        drop_token1 = True

    train_dataset = DataLoader(data1, Batch_size, shuffle=True, drop_last=drop_token1,
                               collate_fn=collate)  
    valid_dataset = DataLoader(data2, Batch_size, shuffle=False, drop_last=False,
                               collate_fn=collate)  

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)  
    stopper = EarlyStopping(mode='higher', patience=patience, tolerance=tolerance,
                            filename=filename)



    # global record
    train_loss_record = []
    valid_loss_record = []
    train_roc_record = []
    valid_roc_record = []
    initial_lr = learning_rate

    
    for turn in range(2):
        if turn ==1 :
            epoch = 0

            # the loss function for contrastive learning
            contras_criterion = ContrastiveLoss()
            while epoch < num_epochs_s//10:
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

                print(
                    f'epoch/num_epochs:{epoch + 1}/{num_epochs_s} ' +
                    f'train_loss_pretrain_epoch: {train_loss_pretrain_epoch} ' + "\n"
                )
                epoch += 1

        epoch = 0

        p = []
        for _ in range(y1.shape[1]):
            yi = y1[:, _]
            p.append(torch.sum(yi == 0) * 1.0 / (torch.sum(yi == 1) + torch.sum(yi == 0)))
        p = torch.FloatTensor(p).to(device)  # 加权交叉熵的权重，只考虑训练集，p.shape=（204，）
        criterion = Newloss(p=p)  # 加权交叉熵初始化，



        while epoch < num_epochs:
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

                # loss for classification task
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
            train_loss_epoch_mean = (torch.sum(train_loss_epoch) / torch.sum(train_n_epoch)).item()  # 按照样本
            train_loss_record.append(train_loss_epoch_mean)
            torch.cuda.empty_cache()  # 清除cuda显存

            train_roc = []
            for _ in range(y1.shape[1]):
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
            for _ in range(y1.shape[1]):
                y_pred = y_pred_valid_epoch[:, _]
                y_true = y_true_valid_epoch[:, _]
                # print(_,np.sum(y_true==1),np.sum(y_true==0))
                valid_roc.append(roc_auc_score(y_true[y_true != -1], y_pred[y_true != -1]))
            valid_roc_record.append(np.mean(valid_roc))

            torch.cuda.empty_cache()
            end = time.perf_counter()
            print("epoch", epoch, "initial_lr", initial_lr)

            print(
                f'epoch/num_epochs:{epoch + 1}/{num_epochs} ' + f"time:{end - start}" + "\n" +
                f'train_loss_epoch: {train_loss_epoch_mean:.5f} ' + f'valid_loss_epoch: {valid_loss_epoch_mean:.5f} ' + "\n" +
                f'train_roc: {np.mean(train_roc):.5f} ' + f' min_train_roc: {min(train_roc):.5f} ' + f' max_train_roc: {max(train_roc):.5f}' + "\n" +
                f'valid_roc: {np.mean(valid_roc):.5f}' + f' min_valid_roc: {min(valid_roc):.5f}' + f' max_valid_roc: {max(valid_roc):.5f}' + "\n"
            )

            early_stop = stopper.step(valid_roc_record[-1], net)
            scheduler.step(np.mean(valid_roc))
            if early_stop:
                break
            epoch = epoch + 1


    df_dict_record = {
        "train_loss_record": train_loss_record,
        "valid_loss_record": valid_loss_record,
        "train_roc_record": train_roc_record,
        "valid_roc_record": valid_roc_record,
    }
    df_record = pd.DataFrame(df_dict_record)
    df_record.to_csv(filename1)
    print("end")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### please specify the location for needed files
    path_i_ori = '/home/ruolan/ipigenetic/my_work/origin_dataset/internal_data/multi_task'
    path_e_ori = '/home/ruolan/ipigenetic/my_work/origin_dataset/external_data/multi_task'
    path_ = '/home/ruolan/ipigenetic/my_work/predictive_result/multi_task/contrast_atom_weight_turn/GCN'
    path_res_record = os.path.join(path_, 'record')
    path_res_superpara = os.path.join(path_, 'superpara')
    path_res_test = os.path.join(path_, 'test')
    path_res_etest = os.path.join(path_, 'etest')

    glist, _ = load_graphs(os.path.join(path_i_ori , 'mul_ori_data.bin'))
    y = np.load(os.path.join(path_i_ori, "y.npy"), allow_pickle=True)

    etest_x , _ = load_graphs(os.path.join(path_e_ori , 'mul_ori_data.bin'))
    y_etest = np.load(os.path.join(os.path.join(path_e_ori, "y.npy")), allow_pickle=True)
    etest_y = torch.from_numpy(y_etest)

    ### record
    data_test = pd.DataFrame()
    data_etest = pd.DataFrame()
    data_superpara = pd.DataFrame()


    for random_seed in range(10):
        init_seeds(seed=random_seed)

        data_test1 = pd.DataFrame()
        data_etest1 = pd.DataFrame()
        skf = StratifiedKFold(n_splits=10, random_state=random_seed, shuffle=True)

        ### data splition
        index = []
        idx = np.arange(y.shape[0])
        for i in range(y.shape[1]):
            index.append([])
            yi = y[:, i]
            idx_i = idx[yi != -1]
            yi = yi[yi != -1]
            for train_set, valid_set in skf.split(idx_i, yi):
                index[i].append(idx_i[valid_set])
        
        y1 = np.ones_like(y) * -1
        y2 = np.ones_like(y1) * -1
        y3 = np.ones_like(y2) * -1
     
        for j in range(10):
            for m in range(y.shape[1]):  
                if j < 8:  
                    y1[:, m][index[m][j]] = y[:, m][index[m][j]]
                elif j == 8:  
                    y2[:, m][index[m][j]] = y[:, m][index[m][j]]
                else:  
                    y3[:, m][index[m][j]] = y[:, m][index[m][j]]
        a1 = np.sum((y1 == -1), axis=1)
        a2 = np.sum((y2 == -1), axis=1)
        a3 = np.sum((y3 == -1), axis=1)
        train_x = [glist[i] for i in range(len(glist)) if a1[i] != (y.shape[1])] 
        valid_x = [glist[i] for i in range(len(glist)) if a2[i] != (y.shape[1])]
        test_x = [glist[i] for i in range(len(glist)) if a3[i] != (y.shape[1])]
        y1 = y1[a1 != y.shape[1]]
        y2 = y2[a2 != y.shape[1]]
        y3 = y3[a3 != y.shape[1]]
        train_y = torch.from_numpy(y1)
        valid_y = torch.from_numpy(y2)
        test_y = torch.from_numpy(y3)

        num_epochs_s = 30
        weight_decay = 1e-4
        ### search for the best hyperparameter combination
        study = optuna.create_study(direction='maximize')
        study.optimize(SelectSuperParameters, n_trials=0)
        trial = study.best_trial
        print("Best hyperparameters: {}".format(trial.params))
        best_hyper = trial.params
        data_superpara = data_superpara.append(pd.DataFrame([best_hyper]))

        ### Train the model with the best hyperparameter combination
        patience = 30
        tolerance = 0.0
        num_epochs = 150
        graph_hidden = best_hyper['graph_hidden']
        depth = best_hyper['depth']
        dropout = best_hyper['dropout']
        learning_rate = best_hyper['learning_rate']
        Batch_size = best_hyper['Batch_size']
        mlp_layers = best_hyper['mlp_layers']

        ### The best hyperparameter combination for AMGC
        # patience = 30
        # tolerance = 0.0
        # num_epochs = 150
        # graph_hidden = 128
        # depth = 3
        # dropout = 0.1
        # learning_rate = 0.0001
        # Batch_size = 32
        # mlp_layers = [1000,200]

        #claim a new network
        net = Classifer(graph_hidden=graph_hidden, depth=depth,
                        mlp_layers=mlp_layers,
                        dropout=dropout).to(device)  
        net.apply(weigth_init)  


        filename = os.path.join(path_res_record , 'mul_task%s.pth'%random_seed)
        filename1 = os.path.join(path_res_record , 'record%s.csv'%random_seed)
        train(net, train_x, valid_x, train_y, valid_y,  learning_rate, weight_decay, Batch_size)

        net1 = torch.load(filename)['model_state_dict']
        test(net1, test_x, test_y, Batch_size)
        test(net1 , etest_x , etest_y , Batch_size)



    #keep the predicted result 
    data_test.index = ['seed', 'mcc' , 'f1' , 'ba' , 'roc' , 'prc']
    data_test = data_test.T
    data_test.to_csv(os.path.join(path_res_test, 'multi_task_test_all.csv') , index = False)
    data_etest.index = ['seed', 'mcc' , 'f1' , 'ba' , 'roc' , 'prc']
    data_etest = data_etest.T
    data_etest.to_csv(os.path.join(path_res_etest, 'multi_task_etest_all.csv') , index = False)
    data_test1.to_csv(os.path.join(path_res_superpara, 'superparameter.csv'), index= False)