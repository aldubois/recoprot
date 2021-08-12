# -*- coding: utf-8 -*-

"""
Tests of the Pdb file reader module.
"""

import os
from .context import recoprot


THIS_DIR = os.path.dirname(__file__)


def test_reader():
    """
    Verify PDB reading file.
    """
    pdb = recoprot.read_pdb(os.path.join(THIS_DIR, "data", "T0759-D1.pdb"))
    atoms = pdb.df["ATOM"]["atom_name"]
    assert atoms[0] == "N"
    assert atoms[1] == "CA"
    return
    
