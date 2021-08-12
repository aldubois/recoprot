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
    structure = recoprot.read_pdb(os.path.join(THIS_DIR, "data", "T0759-D1.pdb"))
    atoms = list(structure.get_atoms())
    assert str(atoms[0]) == "<Atom N>"
    assert str(atoms[1]) == "<Atom CA>"
    return    
