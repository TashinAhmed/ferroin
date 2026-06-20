#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Command-line interface for running the GraphKAN pipeline."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .main import run_pipeline, summarize_predictions
from .parameters import (
    BATCH_SIZE,
    DEFAULT_PRED_DIR,
    DEFAULT_SAVE_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DROPOUT_RATE,
    EPOCHS,
    GRID_SIZE,
    HIDDEN_DIM,
    LAYERS,
    LR,
    PROTEINS,
    SAMPLES_PER_CATEGORY,
    SEED,
    Config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ferroin",
        description="GraphKAN: Graph Kolmogorov-Arnold Network for molecule-protein binding.",
    )
    parser.add_argument("--version", action="version", version=f"ferroin {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument("--train-path", default=DEFAULT_TRAIN_PATH, help="Path to train parquet.")
    data.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to test parquet.")
    data.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Where to save model checkpoints.")
    data.add_argument("--pred-dir", default=DEFAULT_PRED_DIR, help="Where to save predictions.")
    data.add_argument("--proteins", nargs="+", default=list(PROTEINS),
                      help="Protein names to train (default: sEH BRD4 HSA).")
    data.add_argument("--samples-per-category", type=int, default=SAMPLES_PER_CATEGORY,
                      help="Binders / non-binders sampled per protein (training).")
    data.add_argument("--num-test-samples", type=int, default=0,
                      help="If > 0, cap the test set size (quick runs). 0 = full test set.")

    model = parser.add_argument_group("model / training")
    model.add_argument("--epochs", type=int, default=EPOCHS)
    model.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Featurization chunk size.")
    model.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    model.add_argument("--layers", type=int, default=LAYERS)
    model.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    model.add_argument("--lr", type=float, default=LR)
    model.add_argument("--grid-size", type=int, default=GRID_SIZE,
                       help="Fourier grid size for the KAN layers.")
    model.add_argument("--val-frac", type=float, default=0.1,
                       help="Fraction of training data held out for validation (0 disables).")
    model.add_argument("--seed", type=int, default=SEED)
    model.add_argument("--device", default="auto", help="auto | cpu | cuda")

    model.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    model.add_argument("--wandb-project", default="GraphKAN_Protein_Binding")

    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny config (1 protein, few samples, 1 epoch) to verify the install.")
    return parser


def smoke_config() -> Config:
    """A minimal config that exercises the full code path quickly."""
    return Config(
        proteins=["sEH"],
        samples_per_category=200,
        epochs=1,
        layers=2,
        hidden_dim=16,
        grid_size=32,
        batch_size=64,
        infer_batch_size=64,
        val_frac=0.1,
        num_test_samples=200,
    )


def argv_to_config(argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        cfg = smoke_config()
        cfg.train_path = args.train_path
        cfg.test_path = args.test_path
        return cfg
    return Config(
        train_path=args.train_path,
        test_path=args.test_path,
        save_dir=args.save_dir,
        pred_dir=args.pred_dir,
        proteins=args.proteins,
        samples_per_category=args.samples_per_category,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        layers=args.layers,
        dropout_rate=args.dropout,
        lr=args.lr,
        grid_size=args.grid_size,
        seed=args.seed,
        device=args.device,
        val_frac=args.val_frac,
        num_test_samples=args.num_test_samples,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = argv_to_config(argv)
    final_df = run_pipeline(cfg)
    print(summarize_predictions(final_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
