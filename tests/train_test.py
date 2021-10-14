# -*- coding: utf-8 -*-

"""
Test the training of the GNN.
"""

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .context import recoprot

DATA_DIR = "tests/data/diff_file"
PROT_NAME = "1A2K"


class OneProteinDataset(Dataset):

    def __init__(self, x, labels):
        self.x = x
        self.labels = labels
        return

    def __getitem__(self, idx):
        if (idx != 0):
            raise IndexError()
        return self.x, self.labels
        
    def __len__(self):
        return 1
    

def test_train_no_batch():

    x_ref, labels_ref = recoprot.preprocess_ligand_receptor_bound_unbound(
        PROT_NAME, DATA_DIR, 6.)

    x = (
        [torch.from_numpy(x_ref[0][0]),
         torch.from_numpy(x_ref[0][1]),
         torch.from_numpy(x_ref[0][2]),
         torch.from_numpy(x_ref[0][3]),
         torch.from_numpy(x_ref[0][4])],
        [torch.from_numpy(x_ref[1][0]),
         torch.from_numpy(x_ref[1][1]),
         torch.from_numpy(x_ref[1][2]),
         torch.from_numpy(x_ref[1][3]),
         torch.from_numpy(x_ref[1][4])],
    )
    labels = torch.from_numpy(labels_ref)
    
    # Build the dataset and it's loader (no batch)
    dataset = OneProteinDataset(x, labels)

    # Build the GNN

    # Train for 2 epochs
    # Sometimes for some weight values the loss is at 0. and don't move so we repeat
    for _ in range(10):
        gnn = recoprot.CompleteNetwork([128, 256])
        losses = recoprot.train(gnn, dataset, 2)
        if losses[0] != 0.:
            break
    assert losses[0] != losses[1]
    return
