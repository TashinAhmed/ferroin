# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "20/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Test Suite"
# ----------------------------------------------------------------------------

"""
Test function to validate the shape of datasets fetched for specific proteins.
"""

EXPECTED_SHAPES = {
    'sEH': (558142, 6),
    'BRD4': (558859, 6),
    'HSA': (557895, 6)
}


def test_dataset_shapes(datasets, test_datasets):
    """
    Test function to validate dataset shapes for each protein.

    Parameters:
    datasets (dict): Dictionary of training datasets.
    test_datasets (dict): Dictionary of test datasets.
    """
    for protein, expected_shape in EXPECTED_SHAPES.items():
        actual_shape = datasets[protein].shape
        print(f"{protein} dataset shape: ", actual_shape)

        if actual_shape == expected_shape:
            print(f"Full dataset is received for {protein}.")
        else:
            print(f"Full dataset is NOT received for {protein}.")


    print("seh_df test shape: ", test_datasets['sEH'].shape)
    print("brd4_df test shape: ", test_datasets['BRD4'].shape)
    print("hsa_df test shape: ", test_datasets['HSA'].shape)

    if all(datasets[protein].shape == EXPECTED_SHAPES[protein] for protein in datasets):
        print("We now have balanced data for training for each protein, \nwhich we will featurize for our GNN model")