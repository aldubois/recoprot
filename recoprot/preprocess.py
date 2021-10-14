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
    BOUND_LIGAND,
    BOUND_RECEPTOR,
    UNBOUND_LIGAND,
    UNBOUND_RECEPTOR,
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
from .alignment import align_proteins_residues


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
                preprocess_file_and_write_data(os.path.basename(fname), fname, txn, idx=i)

    else:
        with envw.begin(write=True) as txn:
            txn.put(N_PROTEINS.encode(), str(len(options.proteins)).encode())
            for i, pname in enumerate(options.proteins):
                logging.info(
                    f"{i+1}/{len(PROTEINS)} : Preprocessing protein {pname}"
                )
                preprocess_protein_bound_unbound(pname, txn, options.inp, i)
    envw.close()
    return


class Options:

    def __init__(self, inp, out, same_file, db_size, proteins):
        self.inp = inp
        self.out = out
        self.same_file = same_file
        self.db_size = db_size
        self.proteins = (PROTEINS if proteins is None else proteins)
        return

    def __repr__(self):
        return (f"Options(inp={self.inp}, out={self.out},"
                f" same_file={self.same_file},"
                f" db_size={self.db_size},"
                f" proteins={self.proteins})")
    

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
    
    # Optional arguments
    parser.add_argument("-p", "--proteins", dest="proteins",
                        action="store_true",
                        default=None,
                        help="List of proteins to preprocess")
    
    args = parser.parse_args()

    log_fmt = '%(levelname)s: %(message)s'
    if args.log:
        logging.basicConfig(format=log_fmt, level=logging.INFO)
    else:
        logging.basicConfig(format=log_fmt)
    
    return Options(args.inp, args.out, args.same_file, args.db_size, args.proteins)


def preprocess_protein_bound_unbound(protein_name, txn, directory, idx=0, distance=6.):
    
    x, labels = preprocess_ligand_receptor_bound_unbound(
        protein_name,
        directory,
        distance
    )
    verify_data(x, labels)
    write_data(protein_name, x, labels, txn, idx)
    print()
    return


def preprocess_ligand_receptor_bound_unbound(name, folder, distance):
    """
    Preprocessing of ligand and receptor with bound and unbound structures.

    Parameters
    ----------
    name : str
        Protein name.
    folder : str
        Path to the PDB files directory.
    distance : float
        Limit distance for two residues to interact together.

    Returns
    -------
    x : tuple
        Input of the GNN.
    labels : torch.tensor
        Target for the GNN.
    """
    fname_l_b = os.path.join(folder, BOUND_LIGAND.format(name))
    fname_r_b = os.path.join(folder, BOUND_RECEPTOR.format(name))
    fname_l_u = os.path.join(folder, UNBOUND_LIGAND.format(name))
    fname_r_u = os.path.join(folder, UNBOUND_RECEPTOR.format(name))

    chain_l_b, chain_r_b = read_pdb_2prot_different_files(fname_l_b, fname_r_b)
    chain_l_u, chain_r_u = read_pdb_2prot_different_files(fname_l_u, fname_r_u)

    res_l_b = list(chain_l_b.get_residues())
    res_r_b = list(chain_r_b.get_residues())
    res_l_u = list(chain_l_u.get_residues())
    res_r_u = list(chain_r_u.get_residues())

    logging.info(f"    Ligand:")
    res_l_b, res_l_u = align_proteins_residues(res_l_b, res_l_u)

    logging.info(f"    Receptor:")
    res_r_b, res_r_u = align_proteins_residues(res_r_b, res_r_u)

    x = (preprocess_protein(res_l_u), preprocess_protein(res_r_u))
    labels = label_data(res_l_b, res_r_b, distance)
    return x, labels


def preprocess_file_and_write_data(name, filename, txn, idx=0, distance=6.):
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
    x, labels = preprocess_file(filename, distance=distance)
    write_data(name, x, labels, txn, idx)
    return


def write_data(name, x, labels, txn, idx):
    """
    Write data input of the GNN and its labels.
    """
    prefix = f"{idx}"
    txn.put(prefix.encode(), name.encode())
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
    with warnings.catch_warnings(record=True) as w:
        structure2 = parser2.get_structure("", filename2)

    # Patches for problematic pdb files of the DBD5
    if (os.path.basename(filename2).split('_')[0] == "1AZS"):
        chains1 = next(structure1.get_chains())    
        chains2 = list(structure2.get_chains())[1]
    else:
        chains1 = next(structure1.get_chains())    
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
    residues1 = list(chain1.get_residues())
    residues2 = list(chain2.get_residues())
    x = (preprocess_protein(residues1),
         preprocess_protein(residues2))
    target = label_data(residues1, residues2, distance)
    return x, target


def preprocess_protein(residues):
    """
    Preprocess a protein chain to the input data format of the GNN.

    Parameters
    ----------
    residues: list of Bio.PDB.Residue.Residue
        Protein's residues list to consider.

    Returns
    -------
    tuple of np.ndarray
        Arrays containing the atoms encoding, the residues encoding,
        the neighbors encoding and the residue number per atom.
    """
    atoms = [atom for residue in residues for atom in residue.get_atoms()]
    atoms_resname = [atom.get_parent().get_resname() for atom in atoms]
    x_atoms = encode_protein_atoms(atoms).toarray()
    x_residues = encode_protein_residues(atoms_resname).toarray()
    x_same_neigh, x_diff_neigh = encode_neighbors(atoms)        
    residues_names = np.array([i for i, residue in enumerate(residues)
                               for atom in residue.get_atoms()])
    
    if len(set(residues_names)) != len(residues):
        logging.info(set(residues_names))
        logging.info([res.get_id()[1] for res in residues])
    assert len(set(residues_names)) == len(residues)
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
       Atoms list in the proteins (with their positions).
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


def label_data(residues1, residues2, limit=6.):
    """
    Determines if residues from two chains interact with each other.

    Parameters
    ----------
    residues1 : list of Bio.PDB.Residue.Residue
        List of residues to consider in protein1.
    residues2 : list of Bio.PDB.Residue.Residue
        List of residues to consider in protein2.
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
    alpha_carbon = "CA"
    for residue1, residue2 in product(residues1, residues2):
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
    return np.array(labels).astype(np.float32)


def verify_data(x, labels):
    nlabels = len(labels)
    nligand = len(set(x[0][4]))
    nreceptor = len(set(x[1][4]))
    if nlabels != nligand * nreceptor:
        logging.info(f"Labels: {nlabels}")
        logging.info(f"Ligand: {nligand}")
        logging.info(f"Receptor: {nreceptor}")
        logging.info(f"Data: {nligand * nreceptor}")
    assert nlabels == nligand * nreceptor
    return


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
