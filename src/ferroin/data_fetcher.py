#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""DuckDB-based data fetching utilities for the Leash-BIO style parquet data."""

from __future__ import annotations

import pandas as pd


def balanced_data_creation(
    connection,
    file_path: str,
    protein: str,
    samples: int,
    bind_column: str = "binds",
    protein_column: str = "protein_name",
) -> pd.DataFrame:
    """Return a class-balanced DataFrame for one protein (equal binders/non-binders).

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    file_path : str
        Path to the parquet file.
    protein : str
        Protein name to filter on.
    samples : int
        Number of samples to draw from each binding class.
    """
    samples = int(samples)
    query = f"""
    (SELECT * FROM parquet_scan('{file_path}')
     WHERE {bind_column} = 0 AND {protein_column} = '{protein}'
     ORDER BY random()
     LIMIT {samples})
    UNION ALL
    (SELECT * FROM parquet_scan('{file_path}')
     WHERE {bind_column} = 1 AND {protein_column} = '{protein}'
     ORDER BY random()
     LIMIT {samples})
    """
    return connection.query(query).df()


def protein_data(
    connection,
    file_path: str,
    protein: str,
    limit: int | None = None,
    protein_column: str = "protein_name",
) -> pd.DataFrame:
    """Return all rows for a single protein from the parquet file."""
    limit_clause = f"LIMIT {int(limit)}" if limit and int(limit) > 0 else ""
    query = f"""
    SELECT * FROM parquet_scan('{file_path}')
    WHERE {protein_column} = '{protein}'
    {limit_clause}
    """
    return connection.query(query).df()
