# -*- coding: utf-8 -*-

import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset

from .symbols import (
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
    LABELS,
    N_PROTEINS
)


def read_full_db(txn):
    """
    Read all pairs of proteins from the LMDB file.
    
    Parameters
    ----------
    txn : lmdb.Transaction
        LMDB transaction to write in.

    Returns
    -------
    list of tuple containing:
        tuple of tuple of np.ndarray
            Input of the GNN.
        torch.tensor of float
            Target labels of the GNN.
    """
    size = int(txn.get(N_PROTEINS.encode()).decode())
    return [read_protein_pair(txn, idx) for idx in range(size)]


def read_protein_pair(txn, idx):
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
        torch.from_numpy(np.copy(l_residues))
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
        torch.from_numpy(np.copy(r_residues))
    )

    labels = np.frombuffer(txn.get(SEP.join([prefix, LABELS]).encode()), dtype=np.float32)
    labels = torch.from_numpy(np.copy(labels))
    return (x1, x2), labels
    

class ProteinsDataset(Dataset):

    """
    Dataset for proteins pairs.
    """

    def __init__(self, dbpath):
        self.env = lmdb.open(dbpath)
        with self.env.begin(write=False) as txn:
            self.size = int(txn.get(N_PROTEINS.encode()).decode())
        return

    def __getitem__(self, idx):
        if (idx < 0 or idx >= self.size):
            raise IndexError()
        with self.env.begin(write=False) as txn:
            x, y = read_protein_pair(txn, idx)
        return x, y
        
    def __len__(self, value=None):
        return self.size

    def __del__(self):
        self.env.close()
        return
