# -*- coding: utf-8 -*-

"""
Module containing the different neural networks for the experiments.
"""

from itertools import product
from torch import nn


def merge_residues(atoms_per_residue1, atoms_per_residue2):
    """
    Merge the two encoded atoms per residues along the residues cross-product.
    """
    cross_product = product(atoms_per_residue1,
                            atoms_per_residue2)
    return [np.concatenate(pair) for pair in cross_product]



class NoConv(nn.Module):

    """
    Neural network without any convolution layer.
    """

    def __init__(self, ):
        super(Noconv, self).__init__()
        
        return


