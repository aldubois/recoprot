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

class OneProteinDataset(recoprot.ProteinsDataset):
    PROT_NAMES = [PROT_NAME]


def test_complete_network():

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], False)
    preprocess = recoprot.Preprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinDataset("/tmp/test")
    assert(len(loader) == 1)
    name, x, labels = loader[0]

    nn = recoprot.CompleteNetwork([128, 256, 512], [128, 256], False)
    res = nn.forward(x)
    return
