# -*- coding: utf-8 -*-

"""
Module containing the residue level fully connected layers.

This module take as input one line of data for each pair of residue.
"""

# Prerequisites
import torch

# Internal
from ..symbols import DEVICE


class NoConv(torch.nn.Module):

    """
    Neural network without any convolution layer.
    """

    def __init__(self, input_features, layers_sizes):
        """
        Parameters
        ----------
        layers_sizes : list of integer
            Size of each fully connected layers.
        """
        super().__init__()

        # Determine in_features and out_features per layer
        inout_features = [(input_features, layers_sizes[0])]
        for inout_feature in zip(layers_sizes[:-1], layers_sizes[1:]):
            inout_features.append(inout_feature)
        inout_features.append([layers_sizes[-1], 1])

        # Instanciate each
        self.fcs = torch.nn.Sequential(
            *[torch.nn.Linear(in_feature, out_feature, device=DEVICE)
              for in_feature, out_feature in inout_features]
        )


    def forward(self, x):
        """
        Apply each fully connected layers to the
        input data and call softmax on the results.
        """
        for layer in self.fcs[:-1]:
            x = torch.nn.functional.relu(layer(x))
        return self.fcs[-1](x)
