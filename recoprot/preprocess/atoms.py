# -*- coding: utf-8 -*-

"""
Atoms preprocessor.
"""

# Standard Library
import logging

# External Dependencies
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from Bio.PDB.NeighborSearch import NeighborSearch

# Recoprot
from ..symbols import (
    DEVICE,
    SEP,
    ATOMS,
    RESIDUES,
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
    MIN_DISTANCE,
    ALPHA_DISTANCE
)
from .preprocessor import Preprocessor
from .alignment import align_proteins_residues


class AtomsPreprocessor(Preprocessor):

    """
    Preprocessor to generate data at the atom level.
    """

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
        labels : Distances
            Potential distance labels for the GNN.
        """
        res_l_b, res_r_b, res_l_u, res_r_u = AtomsPreprocessor._read_structure(
            folder,
            name
        )

        logging.info("    Ligand:")
        res_l_b, res_l_u = align_proteins_residues(res_l_b, res_l_u)
        logging.info("    Receptor:")
        res_r_b, res_r_u = align_proteins_residues(res_r_b, res_r_u)

        xdata = (
            AtomsPreprocessor._preprocess_protein(res_l_u, tokenizer, model),
            AtomsPreprocessor._preprocess_protein(res_r_u, tokenizer, model)
        )
        labels = AtomsPreprocessor._compute_residues_distance(res_l_b, res_r_b)
        return xdata, labels


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
        x_atoms = AtomsPreprocessor._encode_protein_atoms(atoms).toarray()
        x_residues = (
            Preprocessor._call_protbert(atoms_resname, tokenizer, model)
            if tokenizer is not None and model is not None
            else Preprocessor._encode_protein_residues(atoms_resname).toarray()
        )
        x_same_neigh, x_diff_neigh = AtomsPreprocessor._encode_neighbors(atoms)
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
        txn.put(SEP.join([prefix, MIN_DISTANCE]).encode(), labels.min.tobytes())
        txn.put(SEP.join([prefix, ALPHA_DISTANCE]).encode(), labels.alpha.tobytes())


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
