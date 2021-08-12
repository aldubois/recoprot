# -*- coding: utf-8 -*-

"""
Tests of the preprocessor functions.
"""

import os
from .context import recoprot


THIS_DIR = os.path.dirname(__file__)


def test_pdb2fasta():
    """
    Verify PDB reading file.
    """
    structure = recoprot.read_pdb(os.path.join(THIS_DIR, "data", "T0759-D1.pdb"))
    calc = recoprot.pdb2fasta(structure)
    ref = "VVIHPDPGRELSPEEAHRAGLIDWNMFVKLRSQE"
    assert ref == calc
    return
