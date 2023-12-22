# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/5/14 8:36
@author: Shukai GU
@Filename: multi_task.py
@Software: Vscode
"""

import torch
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import dgl
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

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

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_distance = D(output1, output2)

        loss_contrastive = torch.mean((1 - label) * (1 - cos_distance) +
                                      label * cos_distance)

        return loss_contrastive

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

def metric(data, random_seed , y_pred_train_epoch, y_true_train_epoch):
    if y_pred_train_epoch.shape == y_true_train_epoch.shape:
        mcc = []
        f1 = []
        ba = []
        pre = []
        rec = []
        acc = []
        for _ in range(67):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred])
            y_true = y_true[y_true != -1]
            mcc.append(matthews_corrcoef(y_true, y_pred))
            f1.append(f1_score(y_true, y_pred))
            ba.append(balanced_accuracy_score(y_true, y_pred))
            pre.append(precision_score(y_true, y_pred))
            rec.append(recall_score(y_true, y_pred))
            acc.append(accuracy_score(y_true, y_pred))

        mcc_mean = np.mean(mcc)
        f1_mean = np.mean(f1)
        ba_mean = np.mean(ba)
        pre_mean = np.mean(pre)
        rec_mean = np.mean(rec)
        acc_mean = np.mean(acc)


        roc = []
        prc = []
        for _ in range(67):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_true = y_true[y_true != -1]
            roc.append(roc_auc_score(y_true, y_pred))
            prc1, prc2, _ = precision_recall_curve(y_true, y_pred)
            prc = auc(prc2, prc1)


        roc_mean = np.mean(roc)
        prc_mean = np.mean(prc)
        data['seed%s' % random_seed] = ['seed%s' % random_seed, mcc_mean, f1_mean, ba_mean, roc_mean, prc_mean, pre_mean, rec_mean, acc_mean]
        
    else:
        y_pred_train_epoch = y_pred_train_epoch[:,:36]
        mcc = []
        f1 = []
        ba = []
        pre = []
        rec = []
        acc = []
        for _ in range(36):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred])
            y_true = y_true[y_true != -1]
            mcc.append(matthews_corrcoef(y_true, y_pred))
            f1.append(f1_score(y_true, y_pred))
            ba.append(balanced_accuracy_score(y_true, y_pred))
            pre.append(precision_score(y_true, y_pred))
            rec.append(recall_score(y_true, y_pred))
            acc.append(accuracy_score(y_true, y_pred))
        mcc_mean = np.mean(mcc)
        f1_mean = np.mean(f1)
        ba_mean = np.mean(ba)
        pre_mean = np.mean(pre)
        rec_mean = np.mean(rec)
        acc_mean = np.mean(acc)

        roc = []
        prc = []
        for _ in range(36):
            y_pred = y_pred_train_epoch[:, _]
            y_true = y_true_train_epoch[:, _]
            y_pred = y_pred[y_true != -1]
            y_true = y_true[y_true != -1]
            roc.append(roc_auc_score(y_true, y_pred))
            prc1, prc2, _ = precision_recall_curve(y_true, y_pred)
            prc = auc(prc2, prc1)
        roc_mean = np.mean(roc)
        prc_mean = np.mean(prc)

        data['seed%s' % random_seed] = ['seed%s' % random_seed, mcc_mean, f1_mean, ba_mean, roc_mean, prc_mean,
                                        pre_mean, rec_mean, acc_mean]
        
    return data

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

def test(net, glist3,  y3,  Batch_size, data_test, random_seed=0):
    data3 = list(zip(glist3, y3))
    drop_token1 = False
    # bn>1
    if y3.shape[0] % Batch_size == 1:
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
    
    return metric(data_test,  random_seed, y_pred_test_epoch, y_true_test_epoch)
    
def train(net, glist1, glist2, y1, y2, learning_rate, weight_decay, Batch_size, num_epochs):
    data1 = list(zip(glist1, y1))
    data2 = list(zip(glist2, y2))
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
    stopper = EarlyStopping(mode='higher', patience=30, tolerance=0.0,
                            filename='/home/liuhx/shukai/refer/AMGC/train_models/mul_task_record.pth')
    train_loss_record = []
    valid_loss_record = []
    train_roc_record = []
    valid_roc_record = []
    
    for turn in range(2):
        if turn ==1 :
            epoch = 0
            contras_criterion = ContrastiveLoss()
            while epoch < num_epochs//10:
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
        epoch = 0
        p = []
        for _ in range(y1.shape[1]):
            yi = y1[:, _]
            p.append(torch.sum(yi == 0) * 1.0 / (torch.sum(yi == 1) + torch.sum(yi == 0)))
        p = torch.FloatTensor(p).to(device) 
        criterion = Newloss(p=p)  
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
            valid_loss_epoch_mean = (torch.sum(valid_loss_epoch) / torch.sum(valid_n_epoch)).item() 
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
            # print("epoch", epoch, "initial_lr", initial_lr)
            # print(
            #     f'epoch/num_epochs:{epoch + 1}/{num_epochs} ' + f"time:{end - start}" + "\n" +
            #     f'train_loss_epoch: {train_loss_epoch_mean:.5f} ' + f'valid_loss_epoch: {valid_loss_epoch_mean:.5f} ' + "\n" +
            #     f'train_roc: {np.mean(train_roc):.5f} ' + f' min_train_roc: {min(train_roc):.5f} ' + f' max_train_roc: {max(train_roc):.5f}' + "\n" +
            #     f'valid_roc: {np.mean(valid_roc):.5f}' + f' min_valid_roc: {min(valid_roc):.5f}' + f' max_valid_roc: {max(valid_roc):.5f}' + "\n"
            # )
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
    # df_record.to_csv(filename1)
   
def split_trainingData(glist , y ,random_seed):
    init_seeds(seed=random_seed)
    #Record the results of each target in the testing set and external testing set for each seed
    skf = StratifiedKFold(n_splits=10, random_state=random_seed, shuffle=True)
    # Splitting the internal dataset into training, validation, and testing sets
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
    return train_x,train_y,valid_x,valid_y,test_x,test_y
