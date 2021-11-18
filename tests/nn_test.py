# -*- coding: utf-8 -*-

"""
Test a forward pass of the different Torch neural network.
"""

import lmdb
import numpy as np
import torch

from .context import recoprot


DATA_DIR = "tests/data/diff_file"
PROT_NAME = "1A2K"

class OneProteinAtomsDataset(recoprot.AtomsDataset):
    PROT_NAMES = [PROT_NAME]


class OneProteinResiduesDataset(recoprot.ResiduesDataset):
    PROT_NAMES = [PROT_NAME]

    
def test_atoms_network():

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], False, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinAtomsDataset("/tmp/test", False)
    assert(len(loader) == 1)
    name, x, labels = loader[0]

    nn = recoprot.AtomsNetwork([128, 256, 512], [128, 256], False)
    res = nn.forward(x)
    return


def test_atoms_network_bert():

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], True, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinAtomsDataset("/tmp/test", True)
    assert(len(loader) == 1)
    name, x, labels = loader[0]

    nn = recoprot.AtomsNetwork([128, 256, 512], [128, 256], True)
    res = nn.forward(x)
    return


def test_residues_network():

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], True, False)
    preprocess = recoprot.ResiduesPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinResiduesDataset("/tmp/test", True)
    assert(len(loader) == 1)
    name, x, labels = loader[0]

    nn = recoprot.ResiduesNetwork([128, 256, 512], [128, 256], True)
    res = nn.forward(x)
    return
