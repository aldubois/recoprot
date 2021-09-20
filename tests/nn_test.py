# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import numpy as np
import torch

import lmdb
import recoprot

DATA_FILE = "tests/data/same_file/model.000.00.pdb"


def test_complete_network():

    x, _ = recoprot.preprocess_file(DATA_FILE)
    x = ((torch.from_numpy(x[0][0]),
          torch.from_numpy(x[0][1]),
          torch.from_numpy(x[0][2]),
          torch.from_numpy(x[0][3]),
          x[0][4]),
         (torch.from_numpy(x[1][0]),
          torch.from_numpy(x[1][1]),
          torch.from_numpy(x[1][2]),
          torch.from_numpy(x[1][3]),
          x[1][4])
    )
    nn = recoprot.CompleteNetwork([128, 256])
    res = nn.forward(x)
    return


def test_train():
    x, labels = recoprot.preprocess_file(DATA_FILE, distance=18)
    x = ((torch.from_numpy(x[0][0]),
          torch.from_numpy(x[0][1]),
          torch.from_numpy(x[0][2]),
          torch.from_numpy(x[0][3]),
          x[0][4]),
         (torch.from_numpy(x[1][0]),
          torch.from_numpy(x[1][1]),
          torch.from_numpy(x[1][2]),
          torch.from_numpy(x[1][3]),
          x[1][4])
    )
    labels = torch.from_numpy(labels)
    nn = recoprot.CompleteNetwork([128, 256])
    losses = recoprot.train(nn, x, labels, 2)
    assert losses[0] != losses[1]
    return


def test_read_write_data():

    x_ref, labels_ref = recoprot.preprocess_file(DATA_FILE, distance=18)

    envw = lmdb.open('/tmp/test', max_dbs=2)
    with envw.begin(write=True) as txn:
        recoprot.preprocess_file_and_write_data(DATA_FILE, txn, idx=0, distance=18)
    envw.close()

    envr = lmdb.open('/tmp/test')
    with envr.begin(write=False) as txn:
        x_calc, labels_calc = recoprot.read_input_file(txn, 0)

    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()
    assert (x_ref[0][2] == x_calc[0][2].numpy()).all()
    assert (x_ref[0][3] == x_calc[0][3].numpy()).all()
    assert (x_ref[0][4] == x_calc[0][4]).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()
    assert (x_ref[1][2] == x_calc[1][2].numpy()).all()
    assert (x_ref[1][3] == x_calc[1][3].numpy()).all()
    assert (x_ref[1][4] == x_calc[1][4]).all()

    assert (labels_ref == labels_calc.numpy()).all()
    return
