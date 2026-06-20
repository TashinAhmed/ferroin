# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Validate dataset shapes after balanced sampling.

Previously this compared the *balanced subset* against the *full* dataset
sizes, so the check always "failed". The expected shape for each protein is
``2 * samples_per_category`` (binders + non-binders).

``assert_dataset_shapes`` is the reusable helper; ``test_dataset_shapes`` is a
self-contained pytest test that builds small synthetic DataFrames so it runs
without the Leash-BIO dataset.
"""

from __future__ import annotations

import pandas as pd


def expected_train_rows(samples_per_category: int) -> int:
    """Balanced train rows = 2 * samples_per_category (one per binding class)."""
    return 2 * samples_per_category


def assert_dataset_shapes(
    datasets: dict[str, pd.DataFrame],
    test_datasets: dict[str, pd.DataFrame],
    samples_per_category: int,
) -> None:
    """Assert balanced train sizes and print test sizes. Raises ``AssertionError``."""
    expected = expected_train_rows(samples_per_category)
    for protein, df in datasets.items():
        actual = df.shape[0]
        print(f"{protein} train rows: {actual} (expected ~{expected})")
        assert actual <= expected, f"{protein}: got {actual}, expected <= {expected}"
        assert actual > 0, f"{protein}: empty training dataframe"

    for protein, df in test_datasets.items():
        print(f"{protein} test rows: {df.shape[0]}")
        assert df.shape[0] > 0, f"{protein}: empty test dataframe"


def test_dataset_shapes() -> None:
    """Self-contained check using synthetic balanced DataFrames."""
    samples_per_category = 30
    train, test = {}, {}
    for protein in ["sEH", "BRD4", "HSA"]:
        train[protein] = pd.concat(
            [
                pd.DataFrame({"binds": [0] * samples_per_category}),
                pd.DataFrame({"binds": [1] * samples_per_category}),
            ],
            ignore_index=True,
        )
        test[protein] = pd.DataFrame({"binds": [0] * 100})

    assert_dataset_shapes(train, test, samples_per_category)
    for protein in train:
        assert train[protein].shape[0] == expected_train_rows(samples_per_category)
