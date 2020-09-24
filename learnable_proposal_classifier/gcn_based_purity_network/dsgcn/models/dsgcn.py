#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

__all__ = ['dsgcn']

def eval_batch_new(y, label, loss_type):
        idx = 0
        acc_times = 0.0
        acc_pos_times = 0
        acc_neg_times = 0
        bs = len(y)
        bs_pos = len((label==1).nonzero())
        bs_neg = len((label==0).nonzero())
        if loss_type != 'mse':
            tp_mean_prob = 0
            tp_times = 0
            tn_mean_prob = 0
            tn_times = 0
            fp = 0.0
            fn = 0.0
            fp_iop = 0
            for prob in y:
                this_label = label[idx]
                if prob.cpu() > 0.5 and this_label.cpu() == 1:
                    acc_times += 1
                    acc_pos_times += 1
                    tp_times += 1
                    tp_mean_prob += prob.cpu()
                elif prob.cpu() < 0.5 and this_label.cpu() == 0:
                    acc_times += 1
                    acc_neg_times += 1
                    tn_times += 1
                    tn_mean_prob += prob.cpu()
                elif this_label.cpu() == 1:
                    fn += 1
                elif this_label.cpu() == 0:
                    fp += 1
                    fp_iop += prob.cpu()
                idx += 1
            acc = acc_times / bs
            if bs_pos > 0:
                acc_pos = acc_pos_times / bs_pos
            else:
                acc_pos = 0
            if bs_neg > 0:
                acc_neg = acc_neg_times / bs_neg
            else:
                acc_neg = 0
            if fp != 0:
                fp_iop /= fp
            else:
                fp_iop = 0
            fp = fp / bs
            fn = fn / bs
            if tp_times != 0:
                tp_mean_prob /= tp_times
            else:
                tp_mean_prob = 0
            if tn_times != 0:
                tn_mean_prob /= tn_times
            else:
                tn_mean_prob = 1
            print ('acc = ' + str(acc) + ', acc_pos = ' + str(acc_pos) + ', acc_neg = ' + str(acc_neg) +', fp = ' + str(fp) + ', fn = ' + str(fn) + ', fp_iop = ' + str(fp_iop) + ', tn_mean_prob = ' + str(tn_mean_prob) + ', tp_mean_prob = ' + str(tp_mean_prob))


'''
Original implementation can be referred to:
    - GCN: https://github.com/tkipf/pygcn
'''

def normalize(adj):
    ### TODO  ###
    bs, N, D = adj.shape
    for ii in range(bs):
        sum1 = torch.sum(adj[ii,:,:], dim=1)
        adj[ii,:,:] /= sum1
    return adj

class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, D=None):
        if x.dim() == 3:
            xw = torch.matmul(x, self.weight)
            output = torch.bmm(adj, xw)
        elif x.dim() == 2:
            xw = torch.mm(x, self.weight)
            output = torch.spmm(adj, xw)
        if D is not None:
            output = output * 1. / D
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, freeze_bn, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.gc = GraphConv(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.freeze_bn = freeze_bn
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, adj, index, D=None):
        x = self.gc(x, adj, D)
        bs, N, D = x.shape
        x = x.view(-1, D)
        x1 = x.sum(1)
        index1 = (x1!=0).nonzero()
        index1 = index1.view(-1)
        if self.freeze_bn:
            self.bn.eval()
        x_new = x[index1]
        x_new = self.relu(x_new)
        x[index1] = x_new
        x = x.view(bs, N, D)
        return x

class GNN(nn.Module):
    def __init__(self, planes, feature_dim, featureless,
            num_classes=1, dropout=0.0, reduce_method='max', stage='dev'):
        assert feature_dim > 0
        assert dropout >= 0 and dropout < 1
        if featureless:
            self.inplanes = 1
        else:
            self.inplanes = feature_dim
        self.num_classes = num_classes
        self.reduce_method = reduce_method
        self.feature_dim = feature_dim
        super(GNN, self).__init__()
        if stage == 'dev':
            self.loss = torch.nn.CrossEntropyLoss()
        elif stage == 'seg':
            self.loss = torch.nn.NLLLoss()
        else:
            raise KeyError('Unknown stage: {}'.format(stage))

    def pool(self, x):
        # use global op to reduce N
        # make sure isomorphic graphs output the same representation
        if self.reduce_method == 'sum':
            return torch.sum(x, dim=1)
        elif self.reduce_method == 'mean':
            return torch.mean(x, dim=1)
        elif self.reduce_method == 'max':
            return torch.max(x, dim=1)[0]
        elif self.reduce_method == 'no_pool':
            return x # wo global pooling
        else:
            raise KeyError('Unkown reduce method', self.reduce_method)

    def forward(self, data, return_loss=False):
        purity_label = data[-1].cpu()
        reid_feature = data[0].cpu()
        spatem_feature = data[1].cpu()
        #feature_input = torch.cat([reid_feature, spatem_feature], 2)
        #feature_input = spatem_feature
        feature_input = reid_feature
        x, feature = self.extract(feature_input.cuda(),  data[2].cuda(), data[2].cuda())
        y = self.softmax(x)
        y1 = y[:, 1]
        if return_loss:
            purity_label_test = purity_label.cpu().numpy()
            loss = self.loss(x.view(len(data[-1]),-1), data[-1].long())
            return y, loss
        else:
            eval_batch_new(y1, data[-1], 'BCE')
            return y


class GCN(GNN):

    def __init__(self, planes, feature_dim, featureless,
            num_classes=1, freeze_bn=True, dropout=0.0, reduce_method='max', stage='dev'):
        super().__init__(
                planes, feature_dim, featureless,
                num_classes, dropout, reduce_method, stage)

        self.layers = self._make_layer(BasicBlock, planes, freeze_bn, dropout)
        self.classifier = nn.Linear(self.inplanes, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim = 1)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def _make_layer(self, block, planes, freeze_bn, dropout=0.0):
        layers = nn.ModuleList([])
        for i, plane in enumerate(planes):
            layers.append(block(self.inplanes, plane, freeze_bn, dropout))
            self.inplanes = plane
        return layers

    def extract(self, x, adj, index):
        ### normalize adj ###
        adj = normalize(adj)
        bs = x.size(0)
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        assert (D > 0).all(), "D should larger than 0, otherwise gradient will be NaN."
        for layer in self.layers:
            x = layer(x, adj, index, D)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(-1, self.inplanes)
        feature = x
        x = self.classifier(x)
        if self.reduce_method == 'no_pool':
            if self.num_classes > 1:
                x = x.view(bs, -1, self.num_classes)
                x = torch.transpose(x, 1, 2).contiguous()
                x = F.log_softmax(x, dim=1)
            else:
                x = x.view(bs, -1)
        return x, feature

def _build_model(model_type):
    __model_type__ = {
        'gcn': GCN,
    }
    if model_type not in __model_type__:
        raise KeyError("Unknown model_type:", model_type)
    return __model_type__[model_type]

def dsgcn(feature_dim, hidden_dims=[], featureless=True, \
        gcn_type='gcn', reduce_method='max', dropout=0.5, num_classes=2, freeze_bn=False):
    model = _build_model(gcn_type)
    return model(planes=hidden_dims,
                 feature_dim=feature_dim,
                 featureless=featureless,
                 reduce_method=reduce_method,
                 dropout=dropout,
                 num_classes=num_classes, freeze_bn=freeze_bn)
