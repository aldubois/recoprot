# -*- coding: utf-8 -*-

"""
Module containing the different neural networks for the experiments.
"""

from itertools import product
import numpy as np
import torch
from .preprocess import (
    CATEGORIES,
    ATOMS,
    RESIDUES,
    SEP,
    L_ENC_ATOMS,
    L_ENC_RESIDUES,
    L_NEIGHBORS_IN,
    L_NEIGHBORS_OUT,
    L_RESIDUES,
    R_ENC_ATOMS,
    R_ENC_RESIDUES,
    R_NEIGHBORS_IN,
    R_NEIGHBORS_OUT,
    R_RESIDUES,
    LABELS
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_input_file(txn, idx):
    """
    Read a pair of protein from the LMDB file.
    
    Parameters
    ----------
    txn : lmdb.Transaction
        LMDB transaction to write in.
    idx : int
        Index of the PDB file (ex: you are reading the data from PDB
        file in the environment and you are currently reading the
        idx'th one).

    Returns
    -------
    tuple of tuple of np.ndarray
        Input of the GNN.
    torch.tensor of float
        Target labels of the GNN.
    """
    prefix = f"{idx}"

    l_enc_atoms = np.frombuffer(txn.get(SEP.join([prefix, L_ENC_ATOMS]).encode()), dtype=np.float32)
    l_enc_residues = np.frombuffer(txn.get(SEP.join([prefix, L_ENC_RESIDUES]).encode()), dtype=np.float32)
    l_neighbors_in = np.frombuffer(txn.get(SEP.join([prefix, L_NEIGHBORS_IN]).encode()), dtype=np.int64)
    l_neighbors_out = np.frombuffer(txn.get(SEP.join([prefix, L_NEIGHBORS_OUT]).encode()), dtype=np.int64)
    l_residues = np.frombuffer(txn.get(SEP.join([prefix, L_RESIDUES]).encode()), dtype=np.int64)
    x1 = (
        torch.from_numpy(np.copy(l_enc_atoms.reshape((-1, len(CATEGORIES[ATOMS]))))),
        torch.from_numpy(np.copy(l_enc_residues.reshape((-1, len(CATEGORIES[RESIDUES]))))),
        torch.from_numpy(np.copy(l_neighbors_in.reshape((-1, 10)))),
        torch.from_numpy(np.copy(l_neighbors_out.reshape((-1, 10)))),
        l_residues
    )

    r_enc_atoms = np.frombuffer(txn.get(SEP.join([prefix, R_ENC_ATOMS]).encode()), dtype=np.float32)
    r_enc_residues = np.frombuffer(txn.get(SEP.join([prefix, R_ENC_RESIDUES]).encode()), dtype=np.float32)
    r_neighbors_in = np.frombuffer(txn.get(SEP.join([prefix, R_NEIGHBORS_IN]).encode()), dtype=np.int64)
    r_neighbors_out = np.frombuffer(txn.get(SEP.join([prefix, R_NEIGHBORS_OUT]).encode()), dtype=np.int64)
    r_residues = np.frombuffer(txn.get(SEP.join([prefix, R_RESIDUES]).encode()), dtype=np.int64)
    x2 = (
        torch.from_numpy(np.copy(r_enc_atoms.reshape((-1, len(CATEGORIES[ATOMS]))))),
        torch.from_numpy(np.copy(r_enc_residues.reshape((-1, len(CATEGORIES[RESIDUES]))))),
        torch.from_numpy(np.copy(r_neighbors_in.reshape((-1, 10)))),
        torch.from_numpy(np.copy(r_neighbors_out.reshape((-1, 10)))),
        r_residues
    )

    labels = np.frombuffer(txn.get(SEP.join([prefix, LABELS]).encode()), dtype=np.float32)
    labels = torch.from_numpy(np.copy(labels))
    return (x1, x2), labels
    

def train(network, x, target, n_epoch=10):
    model = network.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    func_loss = torch.nn.BCELoss()
    losses = []
    for epoch in range(1, n_epoch + 1):
        result = model(x)
        loss = func_loss(result, torch.squeeze(target).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:2d}/{n_epoch} -> loss = {loss}")
        losses.append(loss)
        
    return losses


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
        nresidue = len(set(atoms_residue))
        idx = 0
        groups = []
        last_residue = -1
        for residue_id, atom_data in zip(atoms_residue, x):
            if residue_id != last_residue:
                last_residue = residue_id
                groups.append([atom_data])
            else:
                groups[-1].append(atom_data)
        return [torch.stack(group).mean(axis=0) for group in groups]

    
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
