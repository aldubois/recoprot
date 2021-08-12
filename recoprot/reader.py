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
    Bio.PDB.Structure.Structure
        Structure of the protein.
    """
    parser = PDBParser()
    with warnings.catch_warnings(record=True) as w:
        structure = parser.get_structure("", filename)
    return structure
