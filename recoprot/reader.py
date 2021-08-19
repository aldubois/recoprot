# -*- coding: utf-8 -*-

"""
Utility module related to  PDB files parsing.
"""

import warnings

from Bio.PDB.PDBParser import PDBParser


def read_pdb(filename):
    """
    Read a PDB file and output a biopython PDBParser object.

    Parameters
    ----------
    filename: str
        Path to PDB file.

    Returns
    -------
    Bio.PDB.Chain.Chain
        Protein's chain.
    """
    parser = PDBParser()
    with warnings.catch_warnings(record=True) as w:
        structure = parser.get_structure("", filename)
    return next(structure.get_chains())


def read_pdb_two_proteins(filename):
    """
    Read a PDB file and output a biopython PDBParser object.

    Parameters
    ----------
    filename: str
        Path to PDB file.

    Returns
    -------
    Tuple of two Bio.PDB.Chain.Chain
        The two proteins' chains.
    """
    parser = PDBParser()
    with warnings.catch_warnings(record=True) as w:
        structure = parser.get_structure("", filename)
    chains = list(structure.get_chains())
    return chains[0], chains[1]
