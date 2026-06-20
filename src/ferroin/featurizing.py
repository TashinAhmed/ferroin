#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Featurize held-out test data (no labels) into PyG ``Data`` graphs."""

from __future__ import annotations

import pandas as pd
from torch.utils.data import Subset

from .data_creation import featurize_dataframe
from .parameters import INFER_BATCH_SIZE


def featurize_test_df(
    df: pd.DataFrame,
    batch_size: int = INFER_BATCH_SIZE,
    num_samples: int = 0,
) -> list:
    """Featurize a test DataFrame (no labels expected).

    Parameters
    ----------
    df : pandas.DataFrame
        Test set with at least ``molecule_smiles`` and ``id`` columns.
    batch_size : int
        Featurization chunk size.
    num_samples : int
        If > 0, only featurize the first ``num_samples`` rows (useful for a
        quick sanity check). ``0`` uses the entire DataFrame.
    """
    if num_samples and num_samples > 0:
        df = df.iloc[:num_samples]
    data = featurize_dataframe(
        df, batch_size=batch_size, label_column=None, desc="Featurizing test data"
    )
    return data


def slice_data(data, num_samples: int = 10_000):
    """Return a ``Subset`` view of ``data`` limited to ``num_samples`` items."""
    num_samples = min(num_samples, len(data))
    return Subset(data, range(num_samples))
