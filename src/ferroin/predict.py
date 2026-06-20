#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Inference: produce binding probabilities for held-out molecules."""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from .gnn_model import GNNModel


def _to_python(value) -> list:
    """Coerce a collated ``molecule_id`` value into a flat list of Python scalars."""
    if isinstance(value, Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        flat = []
        for v in value:
            flat.extend(_to_python(v))
        return flat
    return [value]


def predict_with_model(
    model: GNNModel,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[list, list[float]]:
    """Return ``(molecule_ids, bind_probabilities)`` for the given loader."""
    model.eval()
    predictions: list[float] = []
    molecule_ids: list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = torch.sigmoid(model(data))
            predictions.extend(output.detach().cpu().view(-1).tolist())
            molecule_ids.extend(_to_python(data.molecule_id))

    return molecule_ids, predictions
