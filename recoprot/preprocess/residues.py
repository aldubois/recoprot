# -*- coding: utf-8 -*-

"""
Residues preprocessor.
"""

# Standard Library
import logging

# External Dependencies
import numpy as np

# Recoprot
from ..symbols import (
    SEP,
    L_ENC_RESIDUES,
    L_NEIGHBORS,
    L_RESIDUES,
    R_ENC_RESIDUES,
    R_NEIGHBORS,
    R_RESIDUES,
    MIN_DISTANCE,
    ALPHA_DISTANCE
)
from .preprocessor import Preprocessor
from .alignment import align_proteins_residues

class ResiduesPreprocessor(Preprocessor):

    """
    Preprocessor of the residue level data.
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
        labels : torch.tensor
            Target for the GNN.
        """
        res_l_b, res_r_b, res_l_u, res_r_u = Preprocessor._read_structure(
            folder,
            name
        )

        logging.info("    Ligand:")
        res_l_b, res_l_u = align_proteins_residues(res_l_b, res_l_u)
        logging.info("    Receptor:")
        res_r_b, res_r_u = align_proteins_residues(res_r_b, res_r_u)

        xdata = (
            ResiduesPreprocessor._preprocess_protein(res_l_u, tokenizer, model),
            ResiduesPreprocessor._preprocess_protein(res_r_u, tokenizer, model)
        )
        labels = Preprocessor._compute_residues_distance(res_l_b, res_r_b)
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
            Arrays containing the residues encoding, and the neighbors encoding.
        """
        residues_name = [residue.get_resname()
                         for residue in residues]
        x_residues = (
            Preprocessor._call_protbert(residues_name, tokenizer, model)
            if tokenizer is not None and model is not None
            else Preprocessor._encode_protein_residues(residues_name).toarray()
        )

        x_neighbors = ResiduesPreprocessor._encode_neighbors(residues)
        xdata = (x_residues.astype(np.float32),
                 x_neighbors)
        return xdata


    @staticmethod
    def _encode_neighbors(residues, n_neighbors=10):

        # Find the 10 closest neighbors for each
        neighbors = []
        for i, residue in enumerate(residues):
            #distances = [ if i != j for j, residue2 in enumerate(residues)]
            distances = [
                (j, Preprocessor._compute_min_distance(residue, residue2))
                for j, residue2 in enumerate(residues) if i != j

            ]
            closest = sorted(distances, key=lambda x : x[1])
            neighbors.append(closest[:min(len(distances), n_neighbors)])


        res = - np.ones((len(residues), n_neighbors), dtype=np.int64)
        for i, data in enumerate(neighbors):
            for j, neighbor in enumerate(data):
                if not np.isinf(neighbor[1]):
                    res[i][j] = neighbor[0]
        return res


    @staticmethod
    def _verify_data(xdata, labels):
        """
        Verify that the dimensions of the data and of the labels match.
        """
        nlabels = len(labels)
        nligand = len(xdata[0][0])
        nreceptor = len(xdata[1][0])
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
        txn.put(SEP.join([prefix, L_ENC_RESIDUES]).encode(),
                xdata[0][0].tobytes())
        txn.put(SEP.join([prefix, L_NEIGHBORS]).encode(),
                xdata[0][1].tobytes())
        # Put receptor protein in file
        txn.put(SEP.join([prefix, R_ENC_RESIDUES]).encode(),
                xdata[1][0].tobytes())
        txn.put(SEP.join([prefix, R_NEIGHBORS]).encode(),
                xdata[1][1].tobytes())
        # Put labels in file
        txn.put(SEP.join([prefix, MIN_DISTANCE]).encode(), labels.min.tobytes())
        txn.put(SEP.join([prefix, ALPHA_DISTANCE]).encode(), labels.alpha.tobytes())
