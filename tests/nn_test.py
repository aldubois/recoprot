# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import numpy as np
import torch
import recoprot


def test_complete_network():

    x, _ = recoprot.preprocess_file("tests/data/model.000.00.pdb")
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
    x, labels = recoprot.preprocess_file("tests/data/model.000.00.pdb", 18)
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
