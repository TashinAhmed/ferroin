#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "20/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "Proof of Concept"
# ----------------------------------------------------------------------------


# Define the path to the dataset file
TRAIN_PATH = './data/train.parquet'

# Define the number of samples per category (binders and non-binders) per protein
SAMPLES_PER_CATEGORY = 30000

# List of proteins to query for
PROTEINS = ['sEH', 'BRD4', 'HSA']

TEST_PATH = './data/test.parquet'

BATCH_SIZE = 2**8
HIDDEN_DIM = 64
EPOCHS = 5
LAYERS = 6
DROPOUT_RATE = 0.3
LR = 0.001

SAVE_DIR = 'trained_models'