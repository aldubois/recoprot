# -*- coding: utf-8 -*-

"""
Test the reading an writing to the LMDB.
"""

import lmdb
import numpy as np
import torch
from .context import recoprot



DATA_DIR = "tests/data/diff_file"
PROT_NAME = "1A2K"

class OneProteinDataset(recoprot.ProteinsDataset):
    PROT_NAMES = [PROT_NAME]
    

def test_read_write_data():

    """
    Test read/write of the LMDB.
    """
    
    x_ref, labels_ref = recoprot.AtomsPreprocessor._preprocess_structure(PROT_NAME, DATA_DIR)

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], False, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinDataset("/tmp/test")
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()
    assert (x_ref[0][2] == x_calc[0][2].numpy()).all()
    assert (x_ref[0][3] == x_calc[0][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[0][4])) == x_calc[0][4]).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()
    assert (x_ref[1][2] == x_calc[1][2].numpy()).all()
    assert (x_ref[1][3] == x_calc[1][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[1][4])) == x_calc[1][4]).all()

    assert ((labels_ref <= 6.) == labels_calc[0].numpy()).all()
    return

