#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""GraphKAN model: Fourier-KAN message passing layers on molecular graphs.

Implements ``NaiveFourierKANLayer`` (Fourier-series Kolmogorov-Arnold layer)
wrapped into a ``MessagePassing`` GNN, followed by global max pooling and a
linear readout. This is the model described in Ahmed & Sifat (2024).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_max_pool

from .featurizer import EDGE_FEATURE_DIM


class NaiveFourierKANLayer(nn.Module):
    """Fourier-series Kolmogorov-Arnold layer.

    Computes, for each output dim ``j``::

        y_j = bias_j + sum_{i,k,d in {cos,sin}} coeffs[d,j,i,k] * trig_d(k * x_i)
    """

    def __init__(self, input_dim: int, out_dim: int, grid_size: int = 300, add_bias: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.out_dim = out_dim

        # [2, out, in, grid] -> index 0 = cos, index 1 = sin.
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_dim, input_dim, grid_size)
            / (np.sqrt(input_dim) * np.sqrt(grid_size))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_dim,)
        x = x.reshape(-1, self.input_dim)

        # k: [1, 1, 1, grid], xrshp: [N, 1, in, 1]
        k = torch.arange(1, self.grid_size + 1, device=x.device).view(1, 1, 1, self.grid_size)
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        c = torch.cos(k * xrshp)  # [N, 1, in, grid]
        s = torch.sin(k * xrshp)

        # Stack cos/sin on a new leading axis -> [2, N, in, grid].
        trig = torch.stack([c.squeeze(1), s.squeeze(1)], dim=0)

        # einsum: d (cos/sin), b (batch), i (in), k (grid) -> bj (batch, out)
        y = torch.einsum("dbik,djik->bj", trig, self.fouriercoeffs)

        if self.add_bias:
            y = y + self.bias
        return y.view(out_shape)


class CustomGNNLayer(MessagePassing):
    """A GraphKAN message-passing layer.

    Messages are built by concatenating the source node features with the edge
    features and passing them through a Fourier-KAN layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = EDGE_FEATURE_DIM,
        grid_size: int = 300,
        add_bias: bool = True,
    ):
        super().__init__(aggr="max")
        self.fourier_kan_layer = NaiveFourierKANLayer(
            in_channels + edge_dim, out_channels, grid_size, add_bias
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat((x_j, edge_attr), dim=1)
        return self.fourier_kan_layer(combined)

    def update(self, aggr_out):
        return aggr_out


class GNNModel(nn.Module):
    """Stack of GraphKAN layers + batch norm + ReLU + dropout + linear readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float,
        edge_dim: int = EDGE_FEATURE_DIM,
        grid_size: int = 300,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [
                CustomGNNLayer(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    edge_dim=edge_dim,
                    grid_size=grid_size,
                )
                for i in range(num_layers)
            ]
        )
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
        return self.lin(x)
