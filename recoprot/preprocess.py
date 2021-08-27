# -*- coding: utf-8 -*-

"""
Data preprocessor.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio.PDB.NeighborSearch import NeighborSearch


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


def preprocess_2_proteins_atoms(chain1, chain2):
    """
    Preprocess the 2 proteins chain to the format of the merge operation.

    For each residue, we encode each of its atoms and then take the average.

    Parameters
    ----------
    chain1: Bio.PDB.Chain.Chain
        Ligand protein's chain.
    chain2: Bio.PDB.Chain.Chain
        Receptor protein's chain.

    Returns
    -------
    list of np.ndarray
        For each residue in the protein chain 1,
        the average of each encoded atoms in this residue.
    list of np.ndarray
        For each residue in the protein chain 2,
        the average of each encoded atoms in this residue.
    """
    residues1 = chain1.get_residues()
    residues2 = chain2.get_residues()
    encoded_atoms1 = [encode_protein_atoms([atom.get_name() for atom in res.get_atoms()]).toarray() for res in residues1]
    encoded_atoms2 = [encode_protein_atoms([atom.get_name() for atom in res.get_atoms()]).toarray() for res in residues2]
    mean_per_residue1 = [atoms.mean(axis=0) for atoms in encoded_atoms1]
    mean_per_residue2 = [atoms.mean(axis=0) for atoms in encoded_atoms2]
    return mean_per_residue1, mean_per_residue2


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


def encode_neighbors(atoms, n_neighbors=10):
    """
    Encode the information of the closest neighbors.

    Parameters
    ----------
    atoms: Iterator of Bio.PDB.Atom.Atom
        Liste d'atomes de la proteine avec leur positions.
    n_neighbors : int
        Number of neighbors to consider.
    """
    # Search neighbors
    searcher = NeighborSearch(atoms)
    found_neighbors = np.array(searcher.search_all(6, "A"))

    # Sort neighbors by distance
    distances = (found_neighbors[:, 0] - found_neighbors[:, 1]).astype(float)
    neighbors = found_neighbors[np.argsort(distances)]

    # Process IDs for matching
    sources, destinations = neighbors[:, 0], neighbors[:, 1]
    atoms_id = np.array([atom.get_serial_number() for atom in atoms]).astype(int)
    residues_id = np.array([atom.get_parent().get_id()[1] for atom in atoms]).astype(int)

    # Initialize the two neighbors list: in the same residue or out of it.
    neighbors_in = - np.ones((len(atoms), n_neighbors), dtype=int)
    neighbors_out = - np.ones((len(atoms), n_neighbors), dtype=int)
    indexes_in = np.zeros(len(atoms), dtype=int)
    indexes_out = np.zeros(len(atoms), dtype=int)

    for src, dest in zip(sources, destinations):

        # Extract ids
        src_atom_id = src.get_serial_number()
        src_residue_id = src.get_parent().get_id()[1]
        dest_atom_id = dest.get_serial_number()
        dest_residue_id = dest.get_parent().get_id()[1]

        # Find the index of the src and destination in the atoms chain
        src_index = np.where(src_atom_id == atoms_id)[0][0]
        dest_index = np.where(dest_atom_id == atoms_id)[0][0]

        # We store the closest neighbors in a numpy array
        
        # Atoms are in the same residues 
        if (src_residue_id == dest_residue_id):
            if (indexes_in[src_index] < n_neighbors):
                neighbors_in[src_index][indexes_in[src_index]] = dest_index
                indexes_in[src_index] += 1
            if (indexes_in[dest_index] < n_neighbors):
                neighbors_in[dest_index][indexes_in[dest_index]] = src_index
                indexes_in[dest_index] += 1

        # Atoms are in different residues
        else:
            if (indexes_out[src_index] < n_neighbors):
                neighbors_out[src_index][indexes_out[src_index]] = dest_index
                indexes_out[src_index] += 1
            if (indexes_out[dest_index] < n_neighbors):
                neighbors_out[dest_index][indexes_out[dest_index]] = src_index
                indexes_out[dest_index] += 1

        
    return neighbors_in, neighbors_out
