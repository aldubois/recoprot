# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import numpy as np
import torch
import recoprot


def test_complete_network():

    x, residues1, residues2, _ = recoprot.preprocess_file("tests/data/model.000.00.pdb")
    nn = recoprot.CompleteNetwork([128, 256], residues1, residues2)
    res = nn.forward(x)
    return
