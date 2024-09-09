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
Data fetching utility functions.
"""

def balanced_data_creation(connection, file_path, protein, samples):
    """
    Fetches a balanced dataset for a specific protein.

    Parameters:
    - connection: DuckDB connection object.
    - file_path: Path to the dataset file.
    - protein: The name of the protein.
    - samples: Number of samples per binder category (1 or 0).

    Returns:
    - A pandas DataFrame containing the balanced dataset for the protein.
    """
    query = f"""
    (SELECT * FROM parquet_scan('{file_path}')
     WHERE binds = 0 AND protein_name = '{protein}'
     ORDER BY random()
     LIMIT {samples})
    UNION ALL
    (SELECT * FROM parquet_scan('{file_path}')
     WHERE binds = 1 AND protein_name = '{protein}'
     ORDER BY random()
     LIMIT {samples})
    """
    return connection.query(query).df()


def protein_data(connection, file_path, protein):
    """
    Fetches the dataset for a specific protein from the provided dataset file.

    Parameters:
    - file_path (str): Path to the dataset file in parquet format.
    - protein (str): The name of the protein to filter the dataset.
    - connection (duckdb.DuckDBPyConnection): An active DuckDB connection object.

    Returns:
    - pandas.DataFrame: A DataFrame containing the subset of the dataset
      where the protein_name matches the specified protein.
    """
    query = f"""
    SELECT * FROM parquet_scan('{file_path}')
    WHERE protein_name = '{protein}'
    """
    return connection.query(query).df()