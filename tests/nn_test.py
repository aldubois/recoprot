# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import numpy as np
import torch
import recoprot


def test_complete_network():

    # Preprocess the data
    chain1, chain2 = recoprot.read_pdb_two_proteins("tests/data/model.000.00.pdb")
    residues1 = np.array([atom.get_parent().get_id()[1]
                          for atom in chain1.get_atoms()])
    residues2 = np.array([atom.get_parent().get_id()[1]
                          for atom in chain2.get_atoms()])
    nn = recoprot.CompleteNetwork([128, 256], residues1, residues2)
    x = (recoprot.preprocess_protein(chain1),
         recoprot.preprocess_protein(chain2))
    res = nn.forward(x)
    return
