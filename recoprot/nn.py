# -*- coding: utf-8 -*-

"""
Module containing the different neural networks for the experiments.
"""

from itertools import product
import numpy as np
from torch import nn
from .preprocess import CATEGORIES


def merge_residues(atoms_per_residue1, atoms_per_residue2):
    """
    Merge the two encoded atoms per residues along the residues cross-product.
    """
    cross_product = product(atoms_per_residue1,
                            atoms_per_residue2)
    return np.array([np.concatenate(pair) for pair in cross_product])



class NoConv(nn.Module):

    """
    Neural network without any convolution layer.
    """

    def __init__(self, layers_sizes):
        """
        Parameters
        ----------
        layers_sizes : list of integer
            Size of each fully connected layers.
        """
        super(NoConv, self).__init__()

        # Determine in_features and out_features per layer
        inout_features = [(2*len(CATEGORIES["atoms"]), layers_sizes[0])]
        for inout_feature in zip(layers_sizes[:-1], layers_sizes[1:]):
            inout_features.append(inout_feature)

        # Instanciate each 
        self.fcs = [
            nn.Linear(in_feature, out_feature)
            for in_feature, out_feature in inout_features
        ]
        return


    def forward(self, x):
        """
        Apply each fully connected layers to the
        input data and call softmax on the results.
        """
        for fc in self.fcs:
            x = fc(x)
        output = x
        # output = nn.functional.log_softmax(x, dim=1)
        return output
