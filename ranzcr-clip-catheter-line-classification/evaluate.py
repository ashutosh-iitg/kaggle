"""Evaluates the model"""

import logging
import os

import numpy as np
import torch
import utils

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        data_batch = data_batch.to(device)
        labels_batch = c1_labels_batch.to(device)
        
        # compute model output and loss
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from tensors, move to cpu, convert to numpy arrays
        output_batch = output_batch.detach().to(torch.device('cpu')).numpy()
        labels_batch = labels_batch.detach().to(torch.device('cpu')).numpy()
        
        # compute all metrics on this batch
        summary_batch = {} #Modify this (Temporary definition to check training)
        #summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
        
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    
    metrics_string = " ; ".join("{}_{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    
    return metrics_mean