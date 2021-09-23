# -*- coding: utf-8 -*-

"""
Module containing the different neural networks for the experiments.
"""

# Python Standard 
from itertools import product

# Prerequisites
import numpy as np
import torch

# Internal
from .preprocess import CATEGORIES, _group_per_residue


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CompleteNetwork(torch.nn.Module):

    """
    Complete neural network.
    """

    def __init__(self, layers_sizes):
        """
        Parameters
        ----------
        layers_sizes : list of integer
            Size of each fully connected layers.
        """
        super(CompleteNetwork, self).__init__()
        self.conv = GNN()
        self.fcs = NoConv(2*self.conv.filters[-1], layers_sizes)
        return

    def forward(self, x):
        # Extract the residue number per atoms from the input
        atoms1_residue = x[0][4]
        atoms2_residue = x[1][4]
        # Call the convolution
        x1, x2 = self.conv.forward(x)
        # Group data and average per residue
        residues1 = self._group_per_residue(atoms1_residue, x1)
        residues2 = self._group_per_residue(atoms2_residue, x2)
        # Concatenate data between the two proteins along the cross product
        cross_product = product(residues1, residues2)
        x3 = torch.stack([torch.cat(pair) for pair in cross_product])
        # Call the fully connected network
        x4 = self.fcs.forward(x3)
        return torch.squeeze(x4)

    @staticmethod
    def _group_per_residue(atoms_residue, x):
        return [tensor.mean(axis=0) for tensor in _group_per_residue(atoms_residue, x)]

    
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
        super(NoConv, self).__init__()

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
        return


    def forward(self, x):
        """
        Apply each fully connected layers to the
        input data and call softmax on the results.
        """
        for fc in self.fcs[:-1]:
            x = torch.nn.functional.relu(fc(x))
        return torch.sigmoid(self.fcs[-1](x))



class GNN_Layer(torch.nn.Module):

    def __init__(self, filters, v_feats, trainable=True, **kwargs):

        super(GNN_Layer, self).__init__()
        self.v_feats = v_feats
        self.filters = filters

        self.trainable = trainable
        self.Wsv = torch.nn.Parameter(torch.randn(self.v_feats, self.filters, device=DEVICE, requires_grad=True))
        self.Wdr = torch.nn.Parameter(torch.randn(self.v_feats, self.filters, device=DEVICE, requires_grad=True))
        self.Wsr = torch.nn.Parameter(torch.randn(self.v_feats, self.filters, device=DEVICE, requires_grad=True))
        self.neighbours = 10
        return

    def forward(self, x):
        return self._forward_one_protein(x[0]), self._forward_one_protein(x[1])

    def _forward_one_protein(self, x):
        Z, same_neigh,diff_neigh = x
        Z = Z.to(DEVICE)
        node_signals = Z @ self.Wsv
        neigh_signals_same=Z @ self.Wsr
        neigh_signals_diff=Z @ self.Wdr
        unsqueezed_same_neigh_indicator = (same_neigh>-1).unsqueeze(2).to(DEVICE)
        unsqueezed_diff_neigh_indicator = (diff_neigh>-1).unsqueeze(2).to(DEVICE)
        same_neigh_features = neigh_signals_same[same_neigh] * unsqueezed_same_neigh_indicator
        diff_neigh_features = neigh_signals_diff[diff_neigh] * unsqueezed_diff_neigh_indicator
        same_norm = torch.sum(same_neigh > -1, 1).unsqueeze(1).type(torch.float).to(DEVICE)
        diff_norm = torch.sum(diff_neigh > -1, 1).unsqueeze(1).type(torch.float).to(DEVICE)

        # To prevent divide by zero error
        same_norm[same_norm==0]=1
        diff_norm[diff_norm==0]=1        
        neigh_same_atoms_signal = (torch.sum(same_neigh_features, axis=1))/same_norm
        neigh_diff_atoms_signal = (torch.sum(diff_neigh_features, axis=1))/diff_norm
        final_res = torch.relu(node_signals +neigh_same_atoms_signal+neigh_diff_atoms_signal)

        return final_res, same_neigh, diff_neigh


class GNN_First_Layer(torch.nn.Module):

    def __init__(self, filters, trainable=True, n_neighbors=10, **kwargs):

        super(GNN_First_Layer, self).__init__()
        self.filters = filters

        self.trainable = trainable
        self.Wv = torch.nn.Parameter(torch.randn(len(CATEGORIES["atoms"]), self.filters, device=DEVICE, requires_grad=True))
        self.Wr = torch.nn.Parameter(torch.randn(len(CATEGORIES["residues"]), self.filters, device=DEVICE,requires_grad=True))
        self.Wsr = torch.nn.Parameter(torch.randn(len(CATEGORIES["atoms"]), self.filters, device=DEVICE, requires_grad=True))
        self.Wdr = torch.nn.Parameter(torch.randn(len(CATEGORIES["atoms"]), self.filters, device=DEVICE, requires_grad=True))
        self.neighbours = n_neighbors

    def forward(self, x):
        return self._forward_one_protein(x[0]), self._forward_one_protein(x[1])
        
    def _forward_one_protein(self, x):
        atoms, residues, same_neigh, diff_neigh, _ = x
        atoms = atoms.to(DEVICE)
        residues = residues.to(DEVICE)
        node_signals = atoms @ self.Wv
        residue_signals = residues @ self.Wr
        neigh_signals_same=atoms @ self.Wsr
        neigh_signals_diff=atoms @ self.Wdr
        unsqueezed_same_neigh_indicator=(same_neigh>-1).unsqueeze(2).to(DEVICE)
        unsqueezed_diff_neigh_indicator=(diff_neigh>-1).unsqueeze(2).to(DEVICE)
        same_neigh_features=neigh_signals_same[same_neigh]*unsqueezed_same_neigh_indicator
        diff_neigh_features=neigh_signals_diff[diff_neigh]*unsqueezed_diff_neigh_indicator
        same_norm = torch.sum(same_neigh > -1, 1).unsqueeze(1).type(torch.float).to(DEVICE)
        diff_norm = torch.sum(diff_neigh > -1, 1).unsqueeze(1).type(torch.float).to(DEVICE)

        # To prevent divide by zero error
        same_norm[same_norm==0] = 1
        diff_norm[diff_norm==0] = 1        
        neigh_same_atoms_signal = (torch.sum(same_neigh_features, axis=1)) / same_norm
        neigh_diff_atoms_signal = (torch.sum(diff_neigh_features, axis=1)) / diff_norm
        
        final_res = torch.relu(node_signals + residue_signals + neigh_same_atoms_signal + neigh_diff_atoms_signal)
        return final_res, same_neigh, diff_neigh
    

class GNN(torch.nn.Module):

    def __init__(self, first_layer_filters=128, other_layers_filters=[256, 512]):
        super(GNN, self).__init__()
        self.filters = [first_layer_filters, *other_layers_filters]
        convs = [GNN_First_Layer(filters=first_layer_filters)]
        if other_layers_filters:
            inout_features = [(first_layer_filters, other_layers_filters[0])]
            for inout_feature in zip(other_layers_filters[:-1], other_layers_filters[1:]):
                inout_features.append(inout_feature)
            for v_feats, filters in inout_features:
                convs.append(GNN_Layer(v_feats=v_feats, filters=filters))
            self.convs = torch.nn.Sequential(*convs)
        return
    
    def forward(self, x):
        for conv in self.convs:
            x = conv.forward(x)
        return x[0][0], x[1][0]
