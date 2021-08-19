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
    fname = os.path.join(THIS_DIR, "data", "model.000.00.pdb")
    chain1, chain2 = recoprot.read_pdb_two_proteins(fname)
    atom1 = next(chain1.get_atoms())
    atom2 = next(chain2.get_atoms())
    assert str(atom1.get_vector()) == "<Vector 9.74, 68.48, 8.62>"
    assert str(atom2.get_vector()) == "<Vector 13.96, 72.53, 7.27>"
    return    
