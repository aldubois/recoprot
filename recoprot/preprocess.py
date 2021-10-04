# -*- coding: utf-8 -*-

"""
Data preprocessor.
"""

# Standard Library
import os
import glob
from itertools import product
import warnings
import argparse
import logging


# External Dependencies
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBParser import PDBParser
import lmdb
import torch

# Recoprot
from .symbols import (
    SEP,
    LIGAND,
    RECEPTOR,
    ENC_ATOMS,
    ENC_RESIDUES,
    NEIGHBORS_IN,
    NEIGHBORS_OUT,
    ATOMS,
    RESIDUES,
    LABELS,
    N_PROTEINS,
    SAME_FILE,
    BOUND_LIGAND_FILE,
    BOUND_RECEPTOR_FILE,
    UNBOUND_LIGAND_FILE,
    UNBOUND_RECEPTOR_FILE,
    L_ENC_ATOMS,
    L_ENC_RESIDUES,
    L_NEIGHBORS_IN,
    L_NEIGHBORS_OUT,
    L_RESIDUES,
    R_ENC_ATOMS,
    R_ENC_RESIDUES,
    R_NEIGHBORS_IN,
    R_NEIGHBORS_OUT,
    R_RESIDUES,
    CATEGORIES,
    RESIDUES_TABLE,
    TRAINING,
    VALIDATION,
    TESTING,
    PROTEINS
)



def preprocess_main():
    options = parse_args()
    preprocess(options)
    return


def preprocess(options):
    envw = lmdb.open(options.out, map_size=options.db_size)

    if options.same_file:
        ftype = os.path.join(options.inp, SAME_FILE)
        files = glob.glob(ftype)
        with envw.begin(write=True) as txn:
            txn.put(N_PROTEINS.encode(), str(len(files)).encode())
            for i, fname in enumerate(files):
                logging.info(f"{i+1}/{len(files)} : Preprocessing "
                             f"file {os.path.basename(fname)}.")
                preprocess_file_and_write_data(fname, txn, idx=i)

    else:
        file_type = os.path.join(options.inp, "*_l_u.pdb")
        prot_names = [
            os.path.basename(fname).split("_")[0]
            for fname in glob.glob(file_type)
        ]
        struct_type = ["b", "u"]
        files = list(product(prot_names, struct_type))
        with envw.begin(write=True) as txn:
            txn.put(N_PROTEINS.encode(), str(len(files)).encode())
            for i, (pname, struct) in enumerate(files):
                fname1 = os.path.join(options.inp, f"{pname}_l_{struct}.pdb")
                fname2 = os.path.join(options.inp, f"{pname}_r_{struct}.pdb")
                logging.info(
                    f"{i+1}/{len(files)} : Preprocessing files "
                    f"{os.path.basename(fname1)} and "
                    f"{os.path.basename(fname2)}."
                )
                preprocess_file_and_write_data(fname1, txn, filename2=fname2, idx=i)

    envw.close()
    return

class Options:

    def __init__(self, inp, out, same_file, db_size):
        self.inp = inp
        self.out = out
        self.same_file = same_file
        self.db_size = db_size
        return

    def __repr__(self):
        return (f"Options(inp={self.inp}, out={self.out},"
                f" same_file={self.same_file},"
                f" db_size={self.db_size})")
    

def parse_args():
    """
    Parse arguments.
    """

    parser = argparse.ArgumentParser(description='Process PDB files.')
 
    # Mandatory arguments
    parser.add_argument("-i", "--input-dir", dest='inp',
                        required=True, metavar='DIR',
                        type=lambda x: _is_dir(parser, x),
                        help='Data output directory')

    parser.add_argument("-o", "--output-dir", dest='out',
                        required=True, metavar='DIR',
                        type=lambda x: _build_dir(parser, x),
                        help='Data output directory')

    # Optional arguments
    parser.add_argument("--same-file", dest="same_file",
                        action="store_true", default=False)

    # Optional arguments
    parser.add_argument("-n", "--db-size", dest="db_size",
                        type=int, default=None,
                        help="Size of the LMDB in Bytes")


    # Optional arguments
    parser.add_argument("--info", dest="log",
                        action="store_true",
                        default=False,
                        help="Display information messages")
    
    args = parser.parse_args()

    log_fmt = '%(levelname)s: %(message)s'
    if args.log:
        logging.basicConfig(format=log_fmt, level=logging.INFO)
    else:
        logging.basicConfig(format=log_fmt)
    
    return Options(args.inp, args.out, args.same_file, args.db_size)


def preprocess_file_and_write_data(filename, txn, filename2=None, idx=0, distance=6.):
    """
    Do the full preprocessing of a file containing 2 proteins.

    The data input of the GNN is generated from the file, the residue
    number per atom for both proteins as well as the data label, that
    will be used as the target of the network. Write this data to a
    LMDB file.

    Parameters
    ----------
    filename : str
        Path to the PDB file to preprocess (containing 2 proteins).
    txn : lmdb.Transaction
        LMDB transaction to write in.
    idx : int
        Index of the PDB file (ex: you are writing 10 PDB file in the
        environment and you are currently writing the idx'th one).
    distance : float
        Distance max for two residues to interact.
    """
    x, labels = (
        preprocess_file(filename, distance=distance)
        if filename2 is None else
        preprocess_file(filename,
                        filename2=filename2,
                        distance=distance)
    )
    prefix = f"{idx}"
    # Put ligand protein in file
    txn.put(SEP.join([prefix, L_ENC_ATOMS]).encode(),
            x[0][0].tobytes())
    txn.put(SEP.join([prefix, L_ENC_RESIDUES]).encode(),
            x[0][1].tobytes())
    txn.put(SEP.join([prefix, L_NEIGHBORS_IN]).encode(),
            x[0][2].tobytes())
    txn.put(SEP.join([prefix, L_NEIGHBORS_OUT]).encode(),
            x[0][3].tobytes())
    txn.put(SEP.join([prefix, L_RESIDUES]).encode(),
            x[0][4].tobytes())
    # Put receptor protein in file
    txn.put(SEP.join([prefix, R_ENC_ATOMS]).encode(),
            x[1][0].tobytes())
    txn.put(SEP.join([prefix, R_ENC_RESIDUES]).encode(),
            x[1][1].tobytes())
    txn.put(SEP.join([prefix, R_NEIGHBORS_IN]).encode(),
            x[1][2].tobytes())
    txn.put(SEP.join([prefix, R_NEIGHBORS_OUT]).encode(),
            x[1][3].tobytes())
    txn.put(SEP.join([prefix, R_RESIDUES]).encode(),
            x[1][4].tobytes())
    # Put labels in file
    txn.put(SEP.join([prefix, LABELS]).encode(), labels.tobytes())
    return
    

def read_pdb_2prot_same_file(filename):
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


def read_pdb_2prot_different_files(filename1, filename2):
    """
    Read a PDB file and output a biopython PDBParser object.

    Parameters
    ----------
    filename1: str
        Path to PDB file for the ligand protein.
    filename2: str
        Path to PDB file for the receptor protein.

    Returns
    -------
    Tuple of two Bio.PDB.Chain.Chain
        The two proteins' chains.
    """
    parser1, parser2 = PDBParser(), PDBParser()
    with warnings.catch_warnings(record=True) as w:
        structure1 = parser1.get_structure("", filename1)
    chains1 = next(structure1.get_chains())
    with warnings.catch_warnings(record=True) as w:
        structure2 = parser2.get_structure("", filename2)
    chains2 = next(structure2.get_chains())
    return chains1, chains2


def preprocess_file(filename, filename2=None, distance=6.):
    """
    Do the full preprocessing of a file containing 2 proteins.
    
    The data input of the GNN is generated from the file, the residue
    number per atom for both proteins as well as the data label, that
    will be used as the target of the network.

    Parameters
    ----------
    filename : str
        Path to the PDB file to preprocess (containing 2 proteins).
    distance : float
        Distance max for two residues to interact.

    Returns
    -------
    tuple of tuple of np.ndarray
        Input of the GNN.
    torch.tensor of float
        Target labels of the GNN.
    """
    chain1, chain2 = (
        read_pdb_2prot_same_file(filename) if filename2 is None
        else read_pdb_2prot_different_files(filename, filename2)
    )
    x = (preprocess_protein(chain1), preprocess_protein(chain2))
    target = label_data(chain1, chain2, distance)
    return x, target


def preprocess_protein(chain):
    """
    Preprocess a protein chain to the input data format of the GNN.

    Parameters
    ----------
    chain: Bio.PDB.Chain.Chain
        Protein's chain.

    Returns
    -------
    tuple of np.ndarray
        Arrays containing the atoms encoding, the residues encoding,
        the neighbors encoding and the residue number per atom.
    """
    atoms = list(chain.get_atoms())
    residues = [atom.get_parent().get_resname() for atom in atoms]
    x_atoms = encode_protein_atoms(atoms).toarray()
    x_residues = encode_protein_residues(residues).toarray()
    x_same_neigh, x_diff_neigh = encode_neighbors(atoms)
    residues_names = np.array([atom.get_parent().get_id()[1]
                               for atom in chain.get_atoms()])
    x = (x_atoms.astype(np.float32), x_residues.astype(np.float32),
         x_same_neigh, x_diff_neigh, residues_names)
    return x


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
    categories = np.array(CATEGORIES[ATOMS]).reshape(-1, 1)
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
        List of residue name per atom in a protein.

    Returns
    -------
    {ndarray, sparse matrix} of shape (n_residues, n_encoded_features)
        Encoded atoms chain in a Compressed Sparse Row format.
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    categories = np.array(CATEGORIES[RESIDUES]).reshape(-1, 1)
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


def label_data(chain1, chain2, limit=6.):
    """
    Determines if residues from two chains interact with each other.

    Parameters
    ----------
    chain1: Bio.PDB.Chain.Chain
        Ligand protein's chain.
    chain2: Bio.PDB.Chain.Chain
        Receptor protein's chain.
    limit: float
        Distance limit in Angstrom.

    Returns
    -------
    np.ndarray of float
        For each pair of residue, indicate if the two
        residues interact with each other.
    """
    # Get the residues number per atom
    labels = []
    residues_product = list(product(chain1.get_residues(),
                                    chain2.get_residues()))
    alpha_carbon = "CA"
    for residue1, residue2 in residues_product:
        atom1 = None
        atom2 = None
        for atom in residue1.get_atoms():
            if atom.get_name() == alpha_carbon:
                atom1 = atom
                break
        for atom in residue2.get_atoms():
            if atom.get_name() == alpha_carbon:
                atom2 = atom
                break
        if (atom1 is None) or (atom2 is None):
            labels.append(float(False))
        else:
            labels.append(float(atom1 - atom2 < limit))
    print(len(labels))
    size =len(residues_product) 
    print(size)
    assert(len(labels) == size)
    return np.array(labels).astype(np.float32)


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


def _is_dir(parser, arg):
    if not os.path.isdir(arg):
        parser.error("The input directory %s does not exist!" % arg)
    else:
        return arg

def _build_dir(parser, arg):
    if os.path.isdir(arg):
        parser.error("The output directory already exist!" % arg)
    else:
        os.makedirs(arg)
        return arg


def _group_per_residue(atoms_residue, x):
    nresidue = len(set(atoms_residue))
    idx = 0
    groups = []
    last_residue = -1
    for residue_id, atom_data in zip(atoms_residue, x):
        if residue_id != last_residue:
            last_residue = residue_id
            groups.append([atom_data])
        else:
            groups[-1].append(atom_data)
    return [torch.stack(group) for group in groups]
