# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:48:58 2020

@author: mayn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


## 模型搭建
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task = 'node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
            
        # post-message-passing
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.Dropout(0.25),
                                     nn.Linear(hidden_dim, hidden_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown Task.')
        
        self.dropout = 0.25
        self.num_layers = 3
    
    def build_conv_model(self, input_dim, hidden_dim):
        ## 不同的gnn实现请参考pytorch几何nn模块。
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, hidden_dim)))
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index. data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, trainnig = self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
        
        x = self.post_mp(x)
        
        return emb, F.log_softmax(x, dim = 1)
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)