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


def train(network, dataset, n_epoch, learning_rate):
    """
    Training function for a GNN.

    Parameters
    ----------
    network : torch.nn.Module
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
    model = network.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []

    # We rebalance each pair of proteins positive values
    positive = 1.e-5
    length = 0
    for _, _, ydata in dataset:
        positive += sum(ydata)
        length += len(ydata)

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([length / positive]).to(DEVICE)
    )
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(1, n_epoch + 1):
        logging.info("Epoch %2d/%d", epoch, n_epoch)
        for _, xdata, ydata in dataset:
            loss = train_step(xdata, ydata)
        logging.info("     -> loss = %f", loss)
        losses.append(loss)

    logging.info("Training set: ")
    auc = evaluate(model, dataset)
    logging.info("    AUC: %.4f" % (auc))

    return model


def evaluate(model, dataset):
    """
    Evaluate a model on a given dataset and return the mean AUC.
    """
    aucs = []
    with torch.no_grad():
        for idx, (name, xdata, target) in enumerate(dataset):
            ydata = model.forward(xdata)
            try:
                aucs.append(roc_auc_score(target.numpy(), ydata.numpy()))
            except ValueError:
                logging.warning("    Complex %s discarded because no positive sample." % name)
    return np.array(aucs).mean()


def make_train_step(model, loss_fn, optimizer):
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
        # Computes loss
        loss = loss_fn(yhat, torch.squeeze(ydata).to(DEVICE))
        # Computes gradients
        loss.backward()
        # Updates parameters
        optimizer.step()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step
