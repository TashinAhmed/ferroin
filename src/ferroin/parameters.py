#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Centralized configuration and hyperparameters for the GraphKAN pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

# Resolve paths relative to the project root (two levels up: src/ferroin -> src -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Default paths / hyperparameters (override via the CLI or environment).
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_PATH = os.environ.get(
    "FERROIN_TRAIN_PATH", str(PROJECT_ROOT / "data" / "train.parquet")
)
DEFAULT_TEST_PATH = os.environ.get(
    "FERROIN_TEST_PATH", str(PROJECT_ROOT / "data" / "test.parquet")
)
DEFAULT_SAVE_DIR = os.environ.get(
    "FERROIN_SAVE_DIR", str(PROJECT_ROOT / "trained_models")
)
DEFAULT_PRED_DIR = os.environ.get(
    "FERROIN_PRED_DIR", str(PROJECT_ROOT / "predictions")
)

PROTEINS: list[str] = ["sEH", "BRD4", "HSA"]

SAMPLES_PER_CATEGORY = 30_000
BATCH_SIZE = 2 ** 8
INFER_BATCH_SIZE = 2 ** 8
HIDDEN_DIM = 64
EPOCHS = 5
LAYERS = 6
DROPOUT_RATE = 0.3
LR = 1.0e-3
GRID_SIZE = 300
SEED = 42
NUM_TEST_SAMPLES = 0  # 0 == use the entire test set


@dataclass
class Config:
    """Runtime configuration for a full train + predict run.

    Fields can be overridden from the CLI (see ``ferroin.cli``) or constructed
    directly when importing the package programmatically.
    """

    train_path: str = DEFAULT_TRAIN_PATH
    test_path: str = DEFAULT_TEST_PATH
    save_dir: str = DEFAULT_SAVE_DIR
    pred_dir: str = DEFAULT_PRED_DIR
    proteins: list[str] = field(default_factory=lambda: list(PROTEINS))
    samples_per_category: int = SAMPLES_PER_CATEGORY
    batch_size: int = BATCH_SIZE
    infer_batch_size: int = INFER_BATCH_SIZE
    hidden_dim: int = HIDDEN_DIM
    epochs: int = EPOCHS
    layers: int = LAYERS
    dropout_rate: float = DROPOUT_RATE
    lr: float = LR
    grid_size: int = GRID_SIZE
    seed: int = SEED
    num_test_samples: int = NUM_TEST_SAMPLES
    val_frac: float = 0.1
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str = "GraphKAN_Protein_Binding"

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Return the torch device requested, defaulting to CUDA when available."""
    if device in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
