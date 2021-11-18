# -*- coding: utf-8 -*-

"""
Module containing the different neural networks at the atom level for the experiments.
"""

# Python Standard
from itertools import product

# Prerequisites
import torch

# Internal
from ..symbols import CATEGORIES, DEVICE
from .fcs import NoConv


class ResiduesNetwork(torch.nn.Module):

    def __init__(self, conv_filters, dense_filters, bert):
        """
        Parameters
        ----------
        conv_filters : list of integer
            Size of each convolution layers.
        dense_filters : list of integer
            Size of each fully connected layers.
        bert : bool
            If Bert was used to pretrain the data input.
        """
        super().__init__()
        v_feats = 1024 if bert else len(CATEGORIES["residues"])
        self.conv = ResiduesGnn(v_feats, conv_filters)
        self.fcs = NoConv(2*self.conv.filters[-1], dense_filters)


    def forward(self, xdata):
        xdata1, xdata2 = self.conv.forward(xdata)
        xdata3 = torch.stack([torch.cat(pair) for pair in product(xdata1, xdata2)])
        return torch.squeeze(self.fcs.forward(xdata3))


class ResiduesGnn(torch.nn.Module):

    def __init__(self, v_feats, filters):

        super().__init__()
        self.v_feats = v_feats
        self.filters = filters

        convs = []
        if self.filters:
            convs.append(GnnLayer(v_feats=self.v_feats, filters=self.filters[0]))
            if len(self.filters) >= 1:
                inout = list(zip(self.filters[:-1], self.filters[1:]))
                for v_feats, filts in inout:
                    convs.append(GnnLayer(v_feats=v_feats, filters=filts))
        self.convs = torch.nn.Sequential(*convs)

    def forward(self, xdata):
        """
        Forward function of the module.
        """
        for conv in self.convs:
            xdata = conv.forward(xdata)
        return xdata[0][0], xdata[1][0]



class GnnLayer(torch.nn.Module):

    def __init__(self, v_feats, filters, neighbors=10, trainable=True):

        super().__init__()

        self.v_feats = v_feats
        self.filters = filters
        self.trainable = trainable
        self.neighbors = neighbors

        # Residue weight matrix
        self.Wr = torch.nn.Parameter(
            torch.randn(
                v_feats,
                filters,
                device=DEVICE,
                requires_grad=True)
        )
        # Residue neighbors weight matrix
        self.Wnr = torch.nn.Parameter(
            torch.randn(
                v_feats,
                filters,
                device=DEVICE,
                requires_grad=True)
        )
        return

    def forward(self, x):
        return self._forward_protein(x[0]), self._forward_protein(x[1])

    def _forward_protein(self, x):
        Z, neighbors = x
        Z = Z.to(DEVICE)

        # Compute signals
        residues_signal = Z @ self.Wr
        neighbors_base_signal = Z @ self.Wnr

        # Apply neighbor signal only to real neighbors and sum
        unsqueezed_neigh_indicator = (neighbors > -1).unsqueeze(2).to(DEVICE)
        neighbors_feats = neighbors_base_signal[neighbors] * unsqueezed_neigh_indicator
        norm = torch.sum(neighbors > -1, 1).unsqueeze(1).type(torch.float).to(DEVICE)
        # To prevent divide by zero error
        norm[norm == 0] = 1
        neighbors_signal = torch.sum(neighbors_feats, axis=1) / norm

        return torch.relu(residues_signal + neighbors_signal), neighbors
