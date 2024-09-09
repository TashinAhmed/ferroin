#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "09/08/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------

# gnn_model.py
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_max_pool
import torch.nn.functional as F
import numpy as np

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, input_dim, out_dim, grid_size=300, add_bias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.out_dim = out_dim

        # Initialize Fourier coefficients
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_dim, input_dim, grid_size) /
            (np.sqrt(input_dim) * np.sqrt(self.grid_size))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        xshp = x.shape
        out_shape = xshp[0:-1] + (self.out_dim,)
        x = x.view(-1, self.input_dim)

        # Create k values for Fourier series (starting from 1)
        k = torch.reshape(torch.arange(1, self.grid_size + 1, device=x.device), (1, 1, 1, self.grid_size))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        # Compute cos and sin components (these should be fused for memory optimization)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # Reshape cos and sin terms for batch processing
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.grid_size))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.grid_size))

        # Use einsum for efficient summing over the Fourier coefficients
        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], axis=0), self.fouriercoeffs)

        if self.add_bias:
            y += self.bias

        y = y.view(out_shape)
        return y

class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, gridsize=300, addbias=True):
        super(CustomGNNLayer, self).__init__(aggr='max')
        self.fourier_kan_layer = NaiveFourierKANLayer(in_channels + 6, out_channels, gridsize, addbias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        transformed = self.fourier_kan_layer(combined)
        return transformed

    def update(self, aggr_out):
        return aggr_out

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([CustomGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_max_pool(x, data.batch)
        x = self.lin(x)
        return x
