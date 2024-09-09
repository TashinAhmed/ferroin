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

"""
Main script to fetch balanced datasets for specific proteins
from a parquet file using DuckDB.
"""

import duckdb
# import pandas as pd

from tests.test_dataset_shapes import test_dataset_shapes
from utils.data_fetcher import balanced_data_creation, protein_data
from utils.parameters import TRAIN_PATH, SAMPLES_PER_CATEGORY, PROTEINS, TEST_PATH

def main():
    """
    Main function to fetch balanced datasets for specific proteins.
    """
    con = duckdb.connect()
    datasets = {}

    for protein in PROTEINS:
        datasets[protein] = balanced_data_creation(
            con, TRAIN_PATH, protein, SAMPLES_PER_CATEGORY
        )

    # Accessing datasets
    seh_df = datasets['sEH']
    brd4_df = datasets['BRD4']
    hsa_df = datasets['HSA']

    test_datasets = {}
    for protein in PROTEINS:
        test_datasets[protein] = protein_data(con, TEST_PATH, protein)

    test_df_seh = test_datasets['sEH']
    test_df_brd4 = test_datasets['BRD4']
    test_df_hsa = test_datasets['HSA']
    con.close()

    test_dataset_shapes(datasets, test_datasets)

if __name__ == "__main__":
    main()
