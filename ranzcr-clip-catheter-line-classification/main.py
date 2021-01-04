import os
import re
import copy
import json
import time
import random
import logging
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import data.dataloader as dataloader
from data.preprocess import DataPreprocess

import models
from models import RANZCRModel
from models import get_score
from train import train_and_evaluate

import utils
from tqdm import tqdm

def main(args):
    # Load the parameters from json file
    params_dir = args.params_dir
    json_path = os.path.join(params_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Set the logger
    model_dir = args.output_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    logging.info("************ Validation fold: {} ************".format(args.fold))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    config_dict = {
        'image_dir': os.path.join(args.input_dir, 'train'),
        'csv_path': os.path.join(args.input_dir, 'train.csv')
    }

    train_data = DataPreprocess(config_dict)
    df, target_cols, num_targets = train_data.df, train_data.target_cols, train_data.num_targets

    # check for debug mode
    if params.debug:
        params.num_epochs = 1
        df = df.sample(n=100, random_state=params.seed).reset_index(drop=True)

    # update params
    params.mode = args.mode
    params.num_targets = num_targets
    params.target_cols = target_cols

    # split data into folds and pass to the model
    Fold = GroupKFold(n_splits=params.num_folds)
    groups = df['PatientID'].values
    for n, (train_index, valid_index) in enumerate(Fold.split(df, df[params.target_cols], groups)):
        df.loc[valid_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)

    # get training and validation data using folds
    train_df = df[df.fold != args.fold].reset_index(drop=True)
    valid_df = df[df.fold == args.fold].reset_index(drop=True)

    # get dataloaders
    train_dataloader = dataloader.fetch_dataloader(train_df, params, data='train')
    valid_dataloader = dataloader.fetch_dataloader(valid_df, params, data='valid')

    logging.info("- done.")

    # Define the model and optimizer
    model = RANZCRModel(params, pretrained=True).model
    if params.cuda:
        model = model.to(torch.device('cuda'))

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # fetch loss function and metrics
    loss_fn = nn.BCEWithLogitsLoss()
    metrics = models.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(
        model, train_dataloader, valid_dataloader,
        optimizer, scheduler, loss_fn, metrics, params, model_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='input', help='input directory')
    parser.add_argument('--params_dir', type=str, default='config', help='directory containing params.json file')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--mode', type=str, default='res', choices=['dense', 'res', 'efficient'], 
                            help='training model')
    parser.add_argument('--fold', type=int, default='4', help='validation fold')

    args = parser.parse_args()
    
    main(args)