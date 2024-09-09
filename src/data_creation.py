#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "21/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------

"""
Main script for creating PyTorch Geometric graph data from SMILES strings and labels,
and for batch featurization.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm

from utils.utils import atom_features, bond_features
from utils.parameters import BATCH_SIZE

featurized_data_train = {}

def create_graph_list(x_smiles, ids, y=None):
    """
    Creates a list of PyTorch Geometric Data objects from SMILES strings and labels.

    Parameters:
    - x_smiles (list): List of SMILES strings representing molecules.
    - ids (list): List of molecule IDs corresponding to the SMILES strings.
    - y (list, optional): List of labels corresponding to the SMILES strings. Default is None.

    Returns:
    - list: A list of PyTorch Geometric Data objects.
    """
    data_list = []

    for index, smiles in enumerate(x_smiles):
        mol = Chem.MolFromSmiles(smiles)

        if not mol:  # Skip invalid SMILES strings
            continue

        # Node features
        atom_features = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_idx = []
        edge_feats = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_idx += [(start, end), (end, start)]  # Undirected graph
            bond_feature = bond_features(bond)
            edge_feats += [bond_feature, bond_feature]  # Same features in both directions

        edge_idx = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        # Creating the Data object
        data = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
        data.molecule_id = ids[index]
        if y is not None:
            data.y = torch.tensor([y[index]], dtype=torch.float)

        data_list.append(data)

    return data_list


def feat_data_in_batches(list_smiles, list_labels, list_IDs, batch_size):
    """
    Featurizes data in batches and returns a list of PyTorch Geometric Data objects.

    Parameters:
    - smiles_list (list): List of SMILES strings representing molecules.
    - labels_list (list): List of labels corresponding to the SMILES strings.
    - ids_list (list): List of molecule IDs corresponding to the SMILES strings.
    - batch_size (int): Size of each batch for featurization.

    Returns:
    - list: A list of PyTorch Geometric Data objects.
    """
    data_list = []

    pbar = tqdm(total=len(list_smiles), desc="Featurizing data")

    for i in range(0, len(list_smiles), batch_size):
        smiles_batch = list_smiles[i: i + batch_size]
        labels_batch = list_labels[i: i + batch_size]
        ids_batch = list_IDs[i: i + batch_size]
        batch_data_list = create_graph_list(
            smiles_batch, ids_batch, labels_batch
        )
        data_list.extend(batch_data_list)
        pbar.update(len(smiles_batch))

    pbar.close()
    return data_list


def main():
    """
    Main function for processing and featurizing the dataset.
    """
    # Define the batch size for featurization
    batch_size = BATCH_SIZE

    # List of proteins and their corresponding dataframes
    proteins_data = {
        "sEH": seh_df,
        "BRD4": brd4_df,
        "HSA": hsa_df,
    }

    # Dictionary to store the featurized data for each protein
    feat_data = {}

    # Loop over each protein and its dataframe
    for protein, df in proteins_data.items():
        print(f"Processing {protein}...")
        smiles_list = df["molecule_smiles"].tolist()
        ids_list = df["id"].tolist()
        labels_list = df["binds"].tolist()

        # Featurize the data
        feat_data[protein] = feat_data_in_batches(
            smiles_list, labels_list, ids_list, batch_size
        )

    train_data_seh = feat_data["sEH"]
    train_data_brd4 = feat_data["BRD4"]
    train_data_hsa = feat_data["HSA"]

    featurized_data_train = {
        'sEH': train_data_seh,
        'BRD4': train_data_brd4,
        'HSA': train_data_hsa
    }

    # Example of saving the featurized data
    torch.save(train_data_seh, "train_data_seh.pt")
    torch.save(train_data_brd4, "train_data_brd4.pt")
    torch.save(train_data_hsa, "train_data_hsa.pt")


if __name__ == "__main__":
    main()
