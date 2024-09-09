#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "20/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------


import numpy as np
from rdkit import Chem
import torch

# from atom_featurizer import one_hot_encoding


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list and returns the one-hot encoded list.

    Parameters:
    - x (str or int): The input element to be one-hot encoded.
    - permitted_list (list): List of permitted values for the one-hot encoding.

    Returns:
    - list: A one-hot encoded list corresponding to the input element.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


"""
Atom featurization utility functions.
"""
def atom_features(atom, use_chirality=True):
    """
    Featurizes an atom based on its type, degree, ring membership, and optionally chirality.

    Parameters:
    - atom (rdkit.Chem.rdchem.Atom): The atom to be featurized.
    - use_chirality (bool): Whether to include chirality in the features. Default is True.

    Returns:
    - numpy.ndarray: A numpy array containing the featurized atom.
    """
    permitted_atom_types = [
        'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Dy', 'Unknown'
    ]
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)

    atom_degree = one_hot_encoding(
        atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour']
    )
    is_in_ring = [int(atom.IsInRing())]

    if use_chirality:
        chirality_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
        )
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring

    return np.array(atom_features, dtype=np.float32)


"""
Bond featurization utility functions.
"""

def bond_features(bond):
    """
    Featurizes a bond based on its type and whether it is in a ring.

    Parameters:
    - bond (rdkit.Chem.rdchem.Bond): The bond to be featurized.

    Returns:
    - numpy.ndarray: A numpy array containing the featurized bond.
    """
    permitted_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'Unknown'
    ]
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'
    features = one_hot_encoding(bond_type, permitted_bond_types) + [int(bond.IsInRing())]
    return np.array(features, dtype=np.float32)


def idx_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def rand_splits(labels, num_classes, trn_percent=0.6, val_percent=0.2):
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
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