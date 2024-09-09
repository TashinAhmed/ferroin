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


# main.py
import torch
from torch_geometric.loader import DataLoader
import wandb
import os
import pandas as pd
from train import train_model
from predict import predict_with_model
from utils.parameters import *
from data_creation import featurized_data_train
from featurizing import test_data_sliced

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = SAVE_DIR
os.makedirs(save_dir, exist_ok=True)
proteins = PROTEINS
all_predictions = []


# wandb.init(project="GNN_Protein_Binding")

for protein in proteins:
    print(f"Training and predicting for {protein}")

    train_loader = DataLoader(featurized_data_train[protein], batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data_sliced[protein], batch_size=16, shuffle=False)

    input_dim = train_loader.dataset[0].num_node_features
    hidden_dim = HIDDEN_DIM
    num_epochs = EPOCHS
    num_layers = LAYERS
    dropout_rate = DROPOUT_RATE
    lr = LR

    model = train_model(train_loader, num_epochs, input_dim, hidden_dim, num_layers, dropout_rate, lr, device)
    model_path = os.path.join(save_dir, f'{protein}_model_{num_epochs}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    molecule_ids, predictions = predict_with_model(model, test_loader, device)

    protein_predictions = pd.DataFrame({
        'id': molecule_ids,
        'binds': predictions,
    })

    all_predictions.append(protein_predictions)
    print(f"Predictions collected for {protein}")

all_predictions_df = pd.concat(all_predictions)
all_predictions_df.to_csv(os.path.join(save_dir, 'all_protein_predictions.csv'), index=False)
print("All predictions saved.")

final_predictions = pd.concat(all_predictions, ignore_index=True)
final_predictions['id'] = final_predictions['id'].apply(lambda x: x.item())
final_predictions.to_csv('final_predictions.csv', index=False)