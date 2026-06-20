#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""End-to-end GraphKAN pipeline: load -> featurize -> train -> predict -> save."""

from __future__ import annotations

import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .data_creation import featurize_dataframe
from .featurizing import featurize_test_df
from .parameters import Config, resolve_device, set_seed
from .predict import predict_with_model
from .preprocessing import load_datasets
from .train import train_model


def run_pipeline(config: Config) -> pd.DataFrame:
    """Run the full train + predict pipeline for every protein in ``config``.

    Returns the concatenated predictions DataFrame (columns: ``id``, ``binds``).
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Using device: {device}")

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.pred_dir, exist_ok=True)

    train_dfs, test_dfs = load_datasets(config)

    all_predictions: list[pd.DataFrame] = []

    for protein in config.proteins:
        print(f"\n=== Training and predicting for {protein} ===")

        train_data = featurize_dataframe(
            train_dfs[protein],
            batch_size=config.batch_size,
            desc=f"Featurizing train [{protein}]",
        )
        test_data = featurize_test_df(
            test_dfs[protein],
            batch_size=config.infer_batch_size,
            num_samples=config.num_test_samples,
        )

        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

        input_dim = train_data[0].num_node_features

        result = train_model(
            dataset=train_data,
            num_epochs=config.epochs,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.layers,
            dropout_rate=config.dropout_rate,
            lr=config.lr,
            device=device,
            batch_size=16,
            grid_size=config.grid_size,
            val_frac=config.val_frac,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )
        model = result["model"]
        print(f"Best score for {protein}: {result['best_score']:.4f}")

        model_path = os.path.join(config.save_dir, f"{protein}_model_{config.epochs}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        molecule_ids, predictions = predict_with_model(model, test_loader, device)

        all_predictions.append(
            pd.DataFrame({"id": molecule_ids, "binds": predictions})
        )
        print(f"Predictions collected for {protein}")

    final_df = pd.concat(all_predictions, ignore_index=True)
    out_path = os.path.join(config.pred_dir, "final_predictions.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\nAll predictions saved to {out_path}")
    return final_df


def summarize_predictions(final_df: pd.DataFrame) -> dict[str, float]:
    """Return simple summary statistics for the prediction DataFrame."""
    return {
        "num_predictions": float(len(final_df)),
        "mean_bind_prob": float(final_df["binds"].mean()),
        "max_bind_prob": float(final_df["binds"].max()),
        "min_bind_prob": float(final_df["binds"].min()),
    }


def main() -> None:
    """Build a default Config and run the pipeline."""
    cfg = Config()
    final_df = run_pipeline(cfg)
    print(summarize_predictions(final_df))


if __name__ == "__main__":
    main()
