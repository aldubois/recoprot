# -*- coding: utf-8 -*-

"""
Test the reading an writing to the LMDB.
"""

import lmdb
import numpy as np
import torch
from .context import recoprot


DATA_FILE = "tests/data/same_file/model.000.00.pdb"

def test_read_write_data():

    x_ref, labels_ref = recoprot.preprocess_file(DATA_FILE, distance=18)

    envw = lmdb.open('/tmp/test', max_dbs=2)
    with envw.begin(write=True) as txn:
        txn.put(recoprot.N_PROTEINS.encode(), str(1).encode())
        recoprot.preprocess_file_and_write_data(DATA_FILE, txn, idx=0, distance=18)
    envw.close()


    loader = recoprot.ProteinsDataset("/tmp/test")
    assert(len(loader) == 1)
    x_calc, labels_calc = loader[0]

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

    assert (labels_ref == labels_calc.numpy()).all()
    return

