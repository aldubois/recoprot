# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import torch
import recoprot


def test_no_conv():

    # Preprocess the data
    chain1, chain2 = recoprot.read_pdb_two_proteins("tests/data/model.000.00.pdb")
    atoms1, atoms2 = recoprot.preprocess_2_proteins_atoms(chain1, chain2)
    res = recoprot.merge_residues(atoms1, atoms2)
    x = torch.from_numpy(res).type(torch.FloatTensor)

    # Create the neural network 
    fcnn = recoprot.NoConv([128, 256])

    # Call the forward pass
    output = fcnn.forward(x)
    assert(output.shape == (res.shape[0], 256))
    return
