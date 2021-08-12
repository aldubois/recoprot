# -*- coding: utf-8 -*-

"""
Data preprocessor.
"""


def pdb2fasta(structure):
    """
    Generate a FASTA sequence from the atom structure.

    Parameters
    ----------
    structure: Bio.PDB.Structure.Structure
        Structure of the protein.

    Returns
    -------
    str
        FASTA sequence.
    """
    # Table extracted from https://cupnet.net/pdb2fasta/
    TABLE = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y"
    }
    return "".join(TABLE[residu.get_resname()] for residu in structure.get_residues())
