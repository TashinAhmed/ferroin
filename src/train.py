#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "09/08/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------


# train.py
import torch
from torch import optim
from torch.nn import BCEWithLogitsLoss
import wandb


def train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate, lr, device):
    from gnn_model import GNNModel  # Import here to avoid circular dependencies

    model = GNNModel(input_dim, hidden_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    # Log optimizer and hyperparameters
    # wandb.watch(model, log='all', log_freq=10)
    # wandb.config.update({
    #     "num_epochs": num_epochs,
    #     "input_dim": input_dim,
    #     "hidden_dim": hidden_dim,
    #     "num_layers": num_layers,
    #     "dropout_rate": dropout_rate,
    #     "lr": lr
    # })

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # wandb.log({"Training Loss": loss.cpu().item(), "Epoch": epoch, "Batch": batch_idx})

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    return model

