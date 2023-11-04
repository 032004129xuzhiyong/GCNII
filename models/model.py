# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月31日
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GCNIILayer
from mytool import tool




class GCNII_model(nn.Module):
    def __init__(self, n_feats, n_class, nlayer, hid_dim, alpha, lamda, dropout, layerclass='GCNIILayer'):
        super().__init__()
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(n_feats, hid_dim))
        GCNIIClass = tool.import_class('models.layers', layerclass)
        for _ in range(nlayer):
            self.convs.append(GCNIIClass(hid_dim, hid_dim))
        self.convs.append(nn.Linear(hid_dim, n_class))

        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list
        avg_adj = torch.stack(adj_list).mean(0)  # [n,n]
        x = torch.concat(x_list, dim=-1)  # [n, allfeatures]
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i, cov in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            beta = math.log(self.lamda/(i+1)+1)
            x = F.relu(cov(x, avg_adj, _hidden[0], self.alpha, beta))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


