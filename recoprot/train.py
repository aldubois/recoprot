# -*- coding: utf-8 -*-

"""
Module to train the GNN.
"""

import logging
import torch

from .nn import DEVICE

def train(network, dataset, n_epoch=10):
    """
    Training function for a GNN.

    Parameters
    ----------
    network : torch.nn.Module
        Model to train.
    dataset : ProteinDataset
        Dataset to work on.
    # dataloader : DataLoader
    #     Dataset to train on.
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
    for _, y in dataset:
        positive += sum(y)
        length += len(y)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([length / positive]))
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(1, n_epoch + 1):
        for i, (x, y) in enumerate(dataset):
            loss = train_step(x, y)
        logging.info(f"Epoch {epoch:2d}/{n_epoch} -> loss = {loss}")
        losses.append(loss)
        
    return losses


def make_train_step(model, loss_fn, optimizer):
    """
    Builds function that performs a step in the train loop.
    
    Extracted from:
    https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#58f2
    """
    def train_step(x, y):
        # Zeroes gradients
        optimizer.zero_grad()
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(yhat, torch.squeeze(y).to(DEVICE))
        # Computes gradients
        loss.backward()
        # Updates parameters
        optimizer.step()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step
