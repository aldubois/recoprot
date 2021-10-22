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
    
    x_ref, labels_ref = recoprot.preprocess_ligand_receptor_bound_unbound(
        PROT_NAME, DATA_DIR, 6.)

    envw = lmdb.open('/tmp/test', max_dbs=2)
    with envw.begin(write=True) as txn:
        txn.put(recoprot.N_PROTEINS.encode(), str(1).encode())
        recoprot.preprocess_protein_bound_unbound(
            PROT_NAME, txn, DATA_DIR,
            idx=recoprot.PROTEINS.index(PROT_NAME)
        )
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

    assert (labels_ref == labels_calc.numpy()).all()
    return

