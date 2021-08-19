# -*- coding: utf-8 -*-

"""
Tests of the preprocessor functions.
"""

import os
from .context import recoprot


THIS_DIR = os.path.dirname(__file__)


def test_pdb2fasta():
    """
    Verify function generating the fasta residue list on the first 10 residues.
    """
    fname = os.path.join(THIS_DIR, "data", "model.000.00.pdb")
    _, chain = recoprot.read_pdb_two_proteins(fname)
    calc = recoprot.pdb2fasta(chain)[:10]
    ref = "MELKNSISDY"
    assert ref == calc
    return
