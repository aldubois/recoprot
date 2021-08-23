# -*- coding: utf-8 -*-

"""
Data preprocessor.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


CATEGORIES = {
    "atoms": ['1', 'C', 'CA', 'CB', 'CG', 'CH2', 'N',
              'NH2', 'O1', 'O2', 'OG','OH', 'SE'],
    "residues": ['1', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
                 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
}

RESIDUES_TABLE = {
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


def pdb2fasta(chain):
    """
    Generate a FASTA sequence from the atom structure.

    Parameters
    ----------
    chain: Bio.PDB.Chain.Chain
        Protein's chain.

    Returns
    -------
    str
        FASTA sequence.
    """
    # Table extracted from https://cupnet.net/pdb2fasta/
    return "".join(RESIDUES_TABLE[residu.get_resname()]
                   for residu in chain.get_residues())


def encode_protein_atoms(atoms):
    """
    Encode protein atoms list into integer array.

    Parameters
    ----------
    atoms: list of str
        List of atoms name from a protein.

    Returns
    -------
    {ndarray, sparse matrix} of shape (n_atoms, n_encoded_features)
        Encoded atoms chain in a Compressed Sparse Row format.
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    categories = np.array(CATEGORIES["atoms"]).reshape(-1, 1)
    encoder.fit(categories)
    categorized_atoms = np.array([atom if atom in categories else '1'
                                  for atom in atoms])
    encoded_atoms = encoder.transform(categorized_atoms.reshape(-1, 1))
    return encoded_atoms


def encode_protein_residues(residues):
    """
    Encode protein residues list into integer array.

    Parameters
    ----------
    residues: list of str
        List of residues name from a protein.

    Returns
    -------
    {ndarray, sparse matrix} of shape (n_residues, n_encoded_features)
        Encoded atoms chain in a Compressed Sparse Row format.
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    categories = np.array(CATEGORIES["residues"]).reshape(-1, 1)
    encoder.fit(categories)
    categorized_residues = np.array([residue if residue in categories else '1'
                                     for residue in residues])
    encoded_residues = encoder.transform(categorized_residues.reshape(-1, 1))
    return encoded_residues
