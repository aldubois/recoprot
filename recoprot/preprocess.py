# -*- coding: utf-8 -*-

"""
Data preprocessor.
"""

# Standard Library
import os
import re
import glob
import requests
from itertools import product
import warnings
import argparse
import logging
from tqdm.auto import tqdm


# External Dependencies
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBParser import PDBParser
import lmdb
import torch
from transformers import BertModel, BertTokenizer

# Recoprot
from .symbols import (
    DEVICE,
    SEP,
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
    PROTEINS
)
from .alignment import align_proteins_residues


def preprocess_main():
    """
    Main function for setuptools entrypoint.
    """
    options = PreprocessorOptions.parse_args()
    preprocess(options)


def preprocess(options):
    """
    Full preprocessing based on user parser options.

    Parameters
    ----------
    options : Options
        Options given by the user through ArgParse.
    """
    preprocess = Preprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            logging.info("%d/%d : Preprocessing protein %s",
                         i + 1, len(PROTEINS), pname)            
            preprocess.preprocess(pname, txn, i)
    envw.close()


class PreprocessorOptions:

    """
    Class containing user-defined options for preprocessing.
    """

    def __init__(self, inp, out, db_size, proteins, protbert):
        self.inp = inp
        self.out = out
        self.db_size = db_size
        self.proteins = (PROTEINS if proteins is None else proteins)
        self.protbert = protbert

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(description='Process PDB files.')
        # Mandatory arguments
        parser.add_argument(
            "-i", "--input-dir", dest='inp',
            required=True, metavar='DIR',
            type=lambda inp: PreprocessorOptions._is_dir(parser, inp),
            help='Data output directory'
        )
        parser.add_argument(
            "-o", "--output-dir", dest='out',
            required=True, metavar='DIR',
            type=lambda inp: PreprocessorOptions._build_dir(parser, inp),
            help='Data output directory'
        )
        # Optional arguments
        parser.add_argument(
            "-n", "--db-size", dest="db_size",
            type=int, default=None,
            help="Size of the LMDB in Bytes"
        )
        # Optional arguments
        parser.add_argument(
            "--info", dest="log",
            action="store_true",
            default=False,
            help="Display information messages"
        )
        # Optional arguments
        parser.add_argument(
            "--protbert", dest="protbert",
            action="store_true",
            default=False,
            help="Display information messages"
        )
        # Optional arguments
        parser.add_argument(
            "-p", "--proteins", dest="proteins",
            action="store_true",
            default=None,
            help="List of proteins to preprocess"
        )
        args = parser.parse_args()
        log_fmt = '%(levelname)s: %(message)s'
        if args.log:
            logging.basicConfig(format=log_fmt, level=logging.INFO)
        else:
            logging.basicConfig(format=log_fmt)
        return cls(args.inp, args.out, args.db_size, args.proteins, args.protbert)
        
    @staticmethod    
    def _is_dir(parser, arg):
        """
        Verify that the argument is an existing directory.
        """
        if not os.path.isdir(arg):
            parser.error(f"The input directory %s does not exist! {arg}")
        return arg

    
    @staticmethod    
    def _build_dir(parser, arg):
        """
        Build a new directory.
        """
        if os.path.isdir(arg):
            parser.error(f"The output directory already exist! {arg}")
        else:
            os.makedirs(arg)
        return arg


    def __repr__(self):
        return (f"Options(inp={self.inp}, out={self.out},"
                f" same_file={self.same_file},"
                f" db_size={self.db_size},"
                f" proteins={self.proteins})")


class Preprocessor:

    def __init__(self, options):
        self.options = options
        if options.protbert:
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False
            )
            self.model = BertModel.from_pretrained(
                "Rostlab/prot_bert"
            )
            self.model = self.model.to(DEVICE)
            self.model = self.model.eval()
        else:
            self.tokenizer = None
            self.model = None
            
    def write_context(self, txn):
        size = len(self.options.proteins)
        txn.put(N_PROTEINS.encode(), str(size).encode())


    def preprocess(self, pname, txn, i):        
        """
        Full processing of a protein pair with both
        its bound and unbound structures.

        Parameters
        ----------
        pname: str
            Name of the protein.
        txn : LMDB transaction object
            Transaction to write the preprocessed data.
        idx: int
            Integer to write a specific entry in the LMDB.
        """
        
        xdata, labels = Preprocessor._preprocess_structure(
            pname,
            self.options.inp,
            self.tokenizer,
            self.model
        )
        Preprocessor._verify_data(xdata, labels)
        Preprocessor._write_data(pname, xdata, labels, txn, i)
        print()

        
    @staticmethod
    def _preprocess_structure(name, folder, tokenizer=None, model=None):
        """
        Preprocessing of ligand and receptor with
        bound and unbound structures.

        Parameters
        ----------
        name : str
            Protein name.
        folder : str
            Path to the PDB files directory.

        Returns
        -------
        xdata : tuple
            Input of the GNN.
        labels : torch.tensor
            Target for the GNN.
        """
        fname_l_b = os.path.join(folder, BOUND_LIGAND.format(name))
        fname_r_b = os.path.join(folder, BOUND_RECEPTOR.format(name))
        fname_l_u = os.path.join(folder, UNBOUND_LIGAND.format(name))
        fname_r_u = os.path.join(folder, UNBOUND_RECEPTOR.format(name))

        chain_l_b, chain_r_b = Preprocessor._read_prot(fname_l_b, fname_r_b)
        chain_l_u, chain_r_u = Preprocessor._read_prot(fname_l_u, fname_r_u)

        res_l_b = list(chain_l_b.get_residues())
        res_r_b = list(chain_r_b.get_residues())
        res_l_u = list(chain_l_u.get_residues())
        res_r_u = list(chain_r_u.get_residues())

        logging.info("    Ligand:")
        res_l_b, res_l_u = align_proteins_residues(res_l_b, res_l_u)
        logging.info("    Receptor:")
        res_r_b, res_r_u = align_proteins_residues(res_r_b, res_r_u)

        xdata = (Preprocessor._preprocess_protein(res_l_u, tokenizer, model),
                 Preprocessor._preprocess_protein(res_r_u, tokenizer, model))
        labels = Preprocessor._compute_alpha_carbon_distance(res_l_b, res_r_b)
        return xdata, labels


    @staticmethod
    def _read_prot(filename1, filename2):
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
        with warnings.catch_warnings(record=True):
            structure1 = parser1.get_structure("", filename1)
        with warnings.catch_warnings(record=True):
            structure2 = parser2.get_structure("", filename2)

        # Patches for problematic pdb files of the DBD5
        if os.path.basename(filename2).split('_')[0] == "1AZS":
            chains1 = next(structure1.get_chains())
            chains2 = list(structure2.get_chains())[1]
        else:
            chains1 = next(structure1.get_chains())
            chains2 = next(structure2.get_chains())
        return chains1, chains2

    
    @staticmethod
    def _preprocess_protein(residues, tokenizer, model):
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
        atoms = [
            atom for residue in residues
            for atom in residue.get_atoms()
        ]
        atoms_resname = [
            atom.get_parent().get_resname()
            for atom in atoms
        ]
        x_atoms = Preprocessor._encode_protein_atoms(atoms).toarray()
        x_residues = (
            Preprocessor._call_protbert(atoms_resname, tokenizer, model)
            if tokenizer is not None and model is not None
            else Preprocessor._encode_protein_residues(atoms_resname).toarray()
        )
        x_same_neigh, x_diff_neigh = Preprocessor._encode_neighbors(atoms)
        residues_names = np.array(
            [i for i, residue in enumerate(residues)
             for atom in residue.get_atoms()]
        )
        if len(set(residues_names)) != len(residues):
            logging.info(set(residues_names))
            logging.info([res.get_id()[1] for res in residues])
        assert len(set(residues_names)) == len(residues)
        xdata = (
            x_atoms.astype(np.float32),
            x_residues.astype(np.float32),
            x_same_neigh,
            x_diff_neigh,
            residues_names
        )
        return xdata


    @staticmethod
    def _encode_protein_atoms(atoms):
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
        categorized_atoms = np.array(
            [atom if atom in categories else '1'
             for atom in atoms])
        encoded_atoms = encoder.transform(
            categorized_atoms.reshape(-1, 1)
        )
        return encoded_atoms


    @staticmethod
    def _encode_protein_residues(residues):
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
        categorized_residues = np.array([
            residue if residue in categories else '1'
            for residue in residues
        ])
        encoded_residues = encoder.transform(
            categorized_residues.reshape(-1, 1)
        )
        return encoded_residues

    @staticmethod
    def _call_protbert(residues, tokenizer, model):
        """
        Create the ProtBert features from the list of residues name.
        
        Parameters
        ----------
        residues: list of str
            List of residue name per atom in a protein.

        Returns
        -------
        np.ndarray : np.ndarray
            ProtBert features.
        """
        categories = np.array(CATEGORIES[RESIDUES]).reshape(-1, 1)
        categorized_residues = " ".join([
            residue if residue in categories else '1'
            for residue in residues
        ])
        ids = tokenizer(
            categorized_residues,
            # add_special_tokens=True,
            # pad_to_max_length=True,
            return_tensors="pt"
        )
        input_ids = ids['input_ids'].to(DEVICE)
        attention_mask = ids['attention_mask'].to(DEVICE)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        return embedding.reshape((embedding.shape[1], embedding.shape[2]))[1:-1]
        

    @staticmethod
    def _compute_alpha_carbon_distance(residues1, residues2):
        # Get the residues number per atom
        distances = []
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
                distances.append(np.Inf)
            else:
                distances.append(atom1 - atom2)
        return np.array(distances).astype(np.float32)
    

    @staticmethod
    def _verify_data(xdata, labels):
        """
        Verify that the dimensions of the data and of the labels match.
        """
        nlabels = len(labels)
        nligand = len(set(xdata[0][4]))
        nreceptor = len(set(xdata[1][4]))
        if nlabels != nligand * nreceptor:
            logging.info("Labels: %d", nlabels)
            logging.info("Ligand: %d", nligand)
            logging.info("Receptor: %d", nreceptor)
            logging.info("Data: %d", nligand * nreceptor)
        assert nlabels == nligand * nreceptor


    @staticmethod
    def _write_data(name, xdata, labels, txn, idx):
        """
        Write data input of the GNN and its labels.
        """
        prefix = f"{idx}"
        txn.put(prefix.encode(), name.encode())
        # Put ligand protein in file
        txn.put(SEP.join([prefix, L_ENC_ATOMS]).encode(),
                xdata[0][0].tobytes())
        txn.put(SEP.join([prefix, L_ENC_RESIDUES]).encode(),
                xdata[0][1].tobytes())
        txn.put(SEP.join([prefix, L_NEIGHBORS_IN]).encode(),
                xdata[0][2].tobytes())
        txn.put(SEP.join([prefix, L_NEIGHBORS_OUT]).encode(),
                xdata[0][3].tobytes())
        txn.put(SEP.join([prefix, L_RESIDUES]).encode(),
                xdata[0][4].tobytes())
        # Put receptor protein in file
        txn.put(SEP.join([prefix, R_ENC_ATOMS]).encode(),
                xdata[1][0].tobytes())
        txn.put(SEP.join([prefix, R_ENC_RESIDUES]).encode(),
                xdata[1][1].tobytes())
        txn.put(SEP.join([prefix, R_NEIGHBORS_IN]).encode(),
                xdata[1][2].tobytes())
        txn.put(SEP.join([prefix, R_NEIGHBORS_OUT]).encode(),
                xdata[1][3].tobytes())
        txn.put(SEP.join([prefix, R_RESIDUES]).encode(),
                xdata[1][4].tobytes())
        # Put labels in file
        txn.put(SEP.join([prefix, LABELS]).encode(), labels.tobytes())


    @staticmethod
    def _encode_neighbors(atoms, n_neighbors=10):
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
        distances = (found_neighbors[:, 0] - found_neighbors[:, 1])
        distances = distances.astype(float)
        neighbors = found_neighbors[np.argsort(distances)]

        # Process IDs for matching
        sources, destinations = neighbors[:, 0], neighbors[:, 1]
        atoms_id = np.array([atom.get_serial_number() for atom in atoms])
        atoms_id = atoms_id.astype(int)

        # Initialize the two neighbors list:
        # in the same residue or out of it.
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
            if src_residue_id == dest_residue_id:
                if indexes_in[src_index] < n_neighbors:
                    neighbors_in[src_index][indexes_in[src_index]] = dest_index
                    indexes_in[src_index] += 1
                if indexes_in[dest_index] < n_neighbors:
                    neighbors_in[dest_index][indexes_in[dest_index]] = src_index
                    indexes_in[dest_index] += 1

            # Atoms are in different residues
            else:
                if indexes_out[src_index] < n_neighbors:
                    neighbors_out[src_index][indexes_out[src_index]] = dest_index
                    indexes_out[src_index] += 1
                if indexes_out[dest_index] < n_neighbors:
                    neighbors_out[dest_index][indexes_out[dest_index]] = src_index
                    indexes_out[dest_index] += 1

        return neighbors_in, neighbors_out
