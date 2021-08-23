# -*- coding: utf-8 -*-

"""
Tests of the preprocessor functions.
"""

import os
import numpy as np
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


def test_encode_protein_atoms():
    """
    Test the encoding of a list of atoms compared to a manually compared one.
    """    
    atoms = ['O2', 'CB', "H", "N", "SE", "O", "CG"]
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    calc = recoprot.encode_protein_atoms(atoms)
    assert (calc.toarray() == ref).all()
    return


def test_encode_protein_residues():
    """
    Test the encoding of a list of residues compared to a manually compared one.
    """
    residues = ['ILE', 'TRP', "ARG", "ZZZ", "GLN", "PHE", "ASN"]
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    calc = recoprot.encode_protein_residues(residues)
    print(calc.toarray())
    print(ref)
    assert (calc.toarray() == ref).all()
    return
