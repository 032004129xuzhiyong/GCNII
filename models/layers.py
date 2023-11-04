# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月31日
"""
import torch
import torch.nn as nn
import math


class GCNIILayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # in_channel == out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, alpha, beta):
        assert x.size(-1) == h0.size(-1)
        left = (1 - alpha) * torch.sparse.mm(adj,x) + alpha * h0
        right = (1-beta) * torch.eye(self.out_channel,device=x.device) + beta * self.weight
        return left @ right


class GCNII_star_Layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # in_channel == out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight1 = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, alpha, beta):
        left = (1-alpha) * torch.sparse.mm(adj,x) @ \
               ((1-beta)*torch.eye(self.out_channel,device=x.device) + beta * self.weight1)
        right = alpha * h0 @ \
                ((1-beta)*torch.eye(self.out_channel,device=x.device) + beta * self.weight2)
        return left + right

if __name__ == '__main__':
    gc = GCNIILayer(32, 32)
    #gc = GCNII_star_Layer(32,32)
    print(gc(torch.rand(50,32), torch.ones(50,50),torch.rand(50,32), 0.1, 0.5).shape)
