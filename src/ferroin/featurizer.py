#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Atom and bond featurization utilities (RDKit -> numeric feature vectors)."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

PERMITTED_ATOM_TYPES = [
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Dy", "Unknown",
]
ATOM_DEGREES = [0, 1, 2, 3, 4, "MoreThanFour"]
CHIRALITY_TAGS = [
    "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER",
]
PERMITTED_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    "Unknown",
]

# Static dimensions, used to build the model without inspecting a sample graph.
ATOM_FEATURE_DIM = len(PERMITTED_ATOM_TYPES) + len(ATOM_DEGREES) + 1 + len(CHIRALITY_TAGS)  # 28
EDGE_FEATURE_DIM = len(PERMITTED_BOND_TYPES) + 1  # 6


def one_hot_encoding(x, permitted_list):
    """One-hot encode ``x`` against ``permitted_list`` (unknown -> last entry)."""
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(bool_value) for bool_value in map(lambda s: x == s, permitted_list)]


def atom_features(atom, use_chirality: bool = True) -> np.ndarray:
    """Featurize an RDKit atom (type, degree, ring membership, chirality)."""
    atom_type = atom.GetSymbol() if atom.GetSymbol() in PERMITTED_ATOM_TYPES else "Unknown"
    feats = one_hot_encoding(atom_type, PERMITTED_ATOM_TYPES)
    feats += one_hot_encoding(atom.GetDegree(), ATOM_DEGREES)
    feats += [int(atom.IsInRing())]
    if use_chirality:
        feats += one_hot_encoding(str(atom.GetChiralTag()), CHIRALITY_TAGS)
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond) -> np.ndarray:
    """Featurize an RDKit bond (type one-hot + ring membership)."""
    bond_type = (
        bond.GetBondType() if bond.GetBondType() in PERMITTED_BOND_TYPES else "Unknown"
    )
    feats = one_hot_encoding(bond_type, PERMITTED_BOND_TYPES)
    feats += [int(bond.IsInRing())]
    return np.asarray(feats, dtype=np.float32)


def idx_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def rand_splits(labels, num_classes, trn_percent: float = 0.6, val_percent: float = 0.2):
    """Class-balanced train/val/test masks (used for node-level splits)."""
    labels = labels.cpu()
    num_classes = num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(labels == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    percls_trn = int(round(trn_percent * (labels.size()[0] / num_classes)))
    val_lb = int(round(val_percent * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = idx_to_mask(train_index, size=labels.size()[0])
    val_mask = idx_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = idx_to_mask(rest_index[val_lb:], size=labels.size()[0])
    return train_mask, val_mask, test_mask
