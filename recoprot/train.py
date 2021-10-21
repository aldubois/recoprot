# -*- coding: utf-8 -*-

"""
Module to train the GNN.
"""

import logging
import torch

from .symbols import DEVICE


def train(network, dataset, n_epoch, learning_rate, limit):
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
    for _, ydata in dataset:
        positive += sum(ydata)
        length += len(ydata)

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([length / positive]).to(DEVICE)
    )
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(1, n_epoch + 1):
        logging.info("Epoch %2d/%d", epoch, n_epoch)
        for xdata, ydata in dataset:
            loss = train_step(xdata, ydata)
        logging.info("     -> loss = %f", loss)
        losses.append(loss)

    success = 0 
    size = 0
    
    with torch.no_grad():
        for xdata, target in dataset:
            yhat = model(xdata)
            calc = (yhat >= limit)
            ref = target.bool()
            success += sum(ref == calc)
            size += len(ref)

    percent = positive * 100. / size
    log.info("Training set: ")
    log.info("    Percentage of cases successfully predicted: %.2f %%" % (percent))

    return model


def evaluate(model, training, validation, cut=0.5):
    """
    Evaluate a model on the training and on the validation set.
    """
    for xdata, target in dataset:
        ydata = model.forward(xdata)
    
    return


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
