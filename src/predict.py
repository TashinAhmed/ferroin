#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "09/09/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------


import torch
from tqdm import tqdm
# import pandas as pd

def predict_with_model(model, test_loader, device):
    model.eval()
    predictions = []
    molecule_ids = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            output = torch.sigmoid(model(data))
            predictions.extend(output.cpu().view(-1).tolist())
            molecule_ids.extend(data.molecule_id.cpu().tolist())

    return molecule_ids, predictions
