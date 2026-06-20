# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Offline smoke tests that do NOT require the Leash-BIO dataset.

They exercise: feature dimensions, graph construction from SMILES, the
GraphKAN forward pass, and the predict helper. Run with ``pytest``.
"""

from __future__ import annotations

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from ferroin.data_creation import create_graph_list, featurize_dataframe
from ferroin.featurizer import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, atom_features
from ferroin.gnn_model import EDGE_FEATURE_DIM as MODEL_EDGE_DIM
from ferroin.gnn_model import GNNModel
from ferroin.parameters import Config, set_seed
from ferroin.predict import predict_with_model


def test_feature_dimensions_consistent():
    """``atom_features`` must produce ``ATOM_FEATURE_DIM`` values."""
    mol = Chem.MolFromSmiles("CCO")
    feats = atom_features(mol.GetAtomWithIdx(0))
    assert len(feats) == ATOM_FEATURE_DIM
    assert MODEL_EDGE_DIM == EDGE_FEATURE_DIM == 6


def test_graph_creation_basic():
    graphs = create_graph_list(["CCO", "c1ccccc1", "not_a_smiles"], ids=[1, 2, 3])
    # The invalid SMILES is skipped -> only 2 graphs.
    assert len(graphs) == 2
    g = graphs[0]
    assert g.x.shape[1] == ATOM_FEATURE_DIM
    assert g.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert g.molecule_id == 1


def test_model_forward_and_predict():
    set_seed(0)
    df = pd.DataFrame(
        {"molecule_smiles": ["CCO", "c1ccccc1", "CCN", "CCC"], "id": [10, 20, 30, 40],
         "binds": [0, 1, 0, 1]}
    )
    data = featurize_dataframe(df, batch_size=4, desc="test")
    loader = DataLoader(data, batch_size=2, shuffle=False)

    model = GNNModel(
        input_dim=ATOM_FEATURE_DIM,
        hidden_dim=8,
        num_layers=2,
        dropout_rate=0.1,
        edge_dim=EDGE_FEATURE_DIM,
        grid_size=16,
    )

    # Forward + backward on one batch.
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == (batch.num_graphs, 1)
    loss = out.sum()
    loss.backward()

    # Predict helper returns aligned ids/probabilities.
    ids, probs = predict_with_model(model, loader, torch.device("cpu"))
    assert len(ids) == len(data)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_config_defaults():
    cfg = Config()
    assert cfg.proteins == ["sEH", "BRD4", "HSA"]
    assert cfg.layers >= 1 and cfg.epochs >= 1
