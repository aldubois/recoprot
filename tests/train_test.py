# -*- coding: utf-8 -*-

"""
Test the training of the GNN.
"""

import lmdb
import numpy as np
import torch
from .context import recoprot


def test_train_no_batch():
    # Build a small LMDB database from test data
    input_dir = "tests/data/diff_file"
    lmdb_dir = "/tmp/data"
    options = recoprot.Options(input_dir, lmdb_dir, False, 50000000)
    recoprot.preprocess(options)

    # Build the dataset and it's loader (no batch)
    dataset = recoprot.ProteinsDataset(lmdb_dir)
    #loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Build the GNN
    gnn = recoprot.CompleteNetwork([128, 256])

    # Train for 2 epochs
    losses = recoprot.train(gnn, dataset, 2)
    assert losses[0] != losses[1]
    return
