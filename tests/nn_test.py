# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import numpy as np
import torch

from .context import recoprot


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
    nn = recoprot.CompleteNetwork([128, 256, 512], [128, 256])
    res = nn.forward(x)
    return
