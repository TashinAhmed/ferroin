#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Fetch balanced training subsets and full test subsets via DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd

from .data_fetcher import balanced_data_creation, protein_data
from .parameters import Config


def load_datasets(
    config: Config,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return ``(train_dfs, test_dfs)`` keyed by protein name.

    Training sets are class-balanced (``samples_per_category`` per class);
    test sets are the full protein subset (no labels expected).
    """
    con = duckdb.connect()
    train_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}

    for protein in config.proteins:
        train_dfs[protein] = balanced_data_creation(
            con, config.train_path, protein, config.samples_per_category
        )
        test_dfs[protein] = protein_data(con, config.test_path, protein)
        print(
            f"[{protein}] train={len(train_dfs[protein])} "
            f"test={len(test_dfs[protein])}"
        )

    con.close()
    return train_dfs, test_dfs


def main() -> None:
    """Standalone entry point: build a Config and print the fetched shapes."""
    cfg = Config()
    train_dfs, test_dfs = load_datasets(cfg)
    for protein in cfg.proteins:
        print(f"{protein}: train={train_dfs[protein].shape}, test={test_dfs[protein].shape}")


if __name__ == "__main__":
    main()
