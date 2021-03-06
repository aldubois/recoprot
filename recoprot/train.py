# -*- coding: utf-8 -*-

"""
Module to train the GNN.
"""

# Python Standard
import logging

# External dependencies
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Internals
from .symbols import DEVICE


def train(model, dataset, n_epoch, learning_rate):
    """
    Training function for a GNN.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    dataset : ProteinDataset
        Dataset to work on.
    n_epoch : int
        Number of epochs.

    Returns
    -------
    losses : list of float
        List of the losses for each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    losses = []

    train_step = make_train_step(model, optimizer)

    for epoch in range(1, n_epoch + 1):
        logging.info("Epoch %2d/%d", epoch, n_epoch)
        for _, xdata, ydata in dataset:
            loss = train_step(xdata, ydata)
        scheduler.step()
        logging.info("     -> loss = %f", loss)
        losses.append(loss)

    return losses


def evaluate(model, dataset):
    """
    Evaluate a model on a given dataset and return the mean AUC.
    """
    aucs = []
    with torch.no_grad():
        for idx, (name, xdata, target) in enumerate(dataset):
            ydata = model.forward(xdata)
            try:
                aucs.append(roc_auc_score(target[0].cpu().numpy(), ydata.cpu().numpy()))
            except ValueError:
                logging.warning("    Complex %s discarded because no positive sample." % name)
    return np.array(aucs).mean() if len(aucs) else np.nan


def make_train_step(model, optimizer):
    """
    Builds function that performs a step in the train loop.

    Extracted from:
    https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#58f2
    """
    def train_step(xdata, ydata):
        # Zeroes gradients
        optimizer.zero_grad()
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(xdata)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            weight=ydata[1].to(DEVICE)
        )
        # Computes loss
        loss = loss_fn(yhat, torch.squeeze(ydata[0]).to(DEVICE))
        # Computes gradients
        loss.backward()
        # Updates parameters
        optimizer.step()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step
