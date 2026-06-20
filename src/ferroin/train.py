#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# ----------------------------------------------------------------------------

"""Training loop for the GraphKAN model with optional validation tracking."""

from __future__ import annotations

import torch
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .gnn_model import GNNModel


def _split_loader(dataset, frac: float, batch_size: int, shuffle: bool):
    """Split ``dataset`` so that a ``frac`` fraction is held out, returning two loaders."""
    n = len(dataset)
    n_val = int(round(frac * n)) if frac and 0 < frac < 1 else 0
    idx = torch.randperm(n).tolist()
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = DataLoader(
        Subset(dataset, train_idx) if n_val else dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_loader = (
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
        if n_val
        else None
    )
    return train_loader, val_loader


def train_model(
    dataset,
    num_epochs: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout_rate: float,
    lr: float,
    device: torch.device,
    batch_size: int = 16,
    edge_dim: int | None = None,
    grid_size: int = 300,
    val_frac: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "GraphKAN_Protein_Binding",
) -> dict[str, object]:
    """Train a GraphKAN model and return a dict with the model + best checkpoint.

    The best model (by validation BCE loss, falling back to training loss when
    no validation split is requested) is loaded back into the returned model.
    """
    if edge_dim is None:
        edge_dim = dataset[0].num_edge_features

    train_loader, val_loader = _split_loader(dataset, val_frac, batch_size, shuffle=True)

    model = GNNModel(
        input_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        edge_dim=edge_dim,
        grid_size=grid_size,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    if use_wandb:
        import wandb

        run = wandb.init(project=wandb_project, config={
            "num_epochs": num_epochs,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "lr": lr,
            "edge_dim": edge_dim,
            "grid_size": grid_size,
        }, reinit=True)
        wandb.watch(model, log="all", log_freq=10)
    else:
        run = None

    best_score = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if run is not None:
                wandb.log(
                    {"train_loss": loss.item(), "epoch": epoch, "batch": batch_idx}
                )
        train_loss /= max(len(train_loader), 1)

        # --- validate ---
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    val_loss += criterion(out, batch.y.view(-1, 1).float()).item()
            val_loss /= max(len(val_loader), 1)

        score = val_loss if val_loss is not None else train_loss
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        msg = f"Epoch {epoch + 1}/{num_epochs}, train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        print(msg)
        if run is not None:
            wandb.log({"epoch_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)

    if run is not None:
        run.finish()

    return {"model": model, "best_score": best_score}
