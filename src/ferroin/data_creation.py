#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Build PyTorch Geometric ``Data`` graphs from SMILES + labels."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from .featurizer import EDGE_FEATURE_DIM, atom_features, bond_features


def create_graph_list(
    x_smiles: Sequence[str],
    ids: Sequence,
    y: Sequence | None = None,
) -> list[Data]:
    """Create a list of PyG ``Data`` objects from SMILES strings and labels.

    Invalid SMILES are skipped. ``ids`` are stored as plain Python values on
    ``data.molecule_id`` (kept consistent with the predict step).
    """
    data_list: list[Data] = []

    for index, smiles in enumerate(x_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # Skip invalid SMILES strings
            continue

        # Node features
        node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(node_feats, dtype=torch.float)

        # Edge index + edge features (undirected -> both directions)
        edge_idx: list[tuple] = []
        edge_feats: list[list] = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_idx += [(start, end), (end, start)]
            bf = bond_features(bond)
            edge_feats += [bf, bf]

        if edge_idx:
            edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        else:  # Single-atom molecules have no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.molecule_id = ids[index]
        if y is not None:
            data.y = torch.tensor([y[index]], dtype=torch.float)

        data_list.append(data)

    return data_list


def feat_data_in_batches(
    list_smiles: Sequence[str],
    list_labels: Sequence | None,
    list_ids: Sequence,
    batch_size: int,
    desc: str = "Featurizing data",
) -> list[Data]:
    """Featurize SMILES in chunks, returning a flat list of ``Data`` objects.

    Note: this is featurization batching (memory-friendly), not training
    batching. The returned list is later wrapped in a PyG ``DataLoader``.
    """
    if list_labels is None:
        list_labels = [-1] * len(list_smiles)

    data_list: list[Data] = []
    pbar = tqdm(total=len(list_smiles), desc=desc)
    for i in range(0, len(list_smiles), batch_size):
        data_list.extend(
            create_graph_list(
                list_smiles[i : i + batch_size],
                list_ids[i : i + batch_size],
                list_labels[i : i + batch_size],
            )
        )
        pbar.update(min(batch_size, len(list_smiles) - i))
    pbar.close()
    return data_list


def featurize_dataframe(
    df: pd.DataFrame,
    batch_size: int,
    smiles_column: str = "molecule_smiles",
    id_column: str = "id",
    label_column: str | None = "binds",
    desc: str = "Featurizing data",
) -> list[Data]:
    """Featurize a DataFrame into a list of PyG ``Data`` objects."""
    smiles_list = df[smiles_column].tolist()
    ids_list = df[id_column].tolist()
    labels_list = df[label_column].tolist() if label_column in df.columns else None
    return feat_data_in_batches(smiles_list, labels_list, ids_list, batch_size, desc=desc)
