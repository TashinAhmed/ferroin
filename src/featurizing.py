#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "22/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------

import pandas as pd
from torch.utils.data import DataLoader, Subset

from src.data_creation import feat_data_in_batches

test_data_sliced = {}


# Assuming featurize_data_in_batches is defined elsewhere and imported

def process_test_data(test_df, batch_size=2 ** 8):
    smiles_list = test_df['molecule_smiles'].tolist()
    ids_list = test_df['id'].tolist()
    labels_list = [-1] * len(smiles_list)
    return feat_data_in_batches(smiles_list, labels_list, batch_size)


def main():
    global test_data_brd4, test_data_seh, test_data_hsa

    # Load your dataframes (assuming they are defined elsewhere)
    brd4_test_df = pd.read_csv('path_to_brd4_test.csv')
    seh_test_df = pd.read_csv('path_to_seh_test.csv')
    hsa_test_df = pd.read_csv('path_to_hsa_test.csv')

    # Process test data for each dataset
    test_data_brd4 = process_test_data(brd4_test_df)
    test_data_seh = process_test_data(seh_test_df)
    test_data_hsa = process_test_data(hsa_test_df)

    featurized_data_test = {
        'sEH': test_data_seh,
        'BRD4': test_data_brd4,
        'HSA': test_data_hsa
    }

    # Print the first element of seh_test_data
    # print(seh_test_data[0])

    def slice_data(data, num_samples=10000):
        return Subset(data, range(num_samples))

    # Slice each dataset to include only the first 10 K samples
    test_data_sliced = {
        protein: slice_data(data, 10000)
        for protein, data in featurized_data_test.items()
    }


if __name__ == "__main__":
    main()
