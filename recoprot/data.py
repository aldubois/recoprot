# -*- coding: utf-8 -*-

"""
Pytorch Datasets for training.
"""

import abc

import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset

from .symbols import (
    CATEGORIES,
    ATOMS,
    RESIDUES,
    SEP,
    L_ENC_ATOMS,
    L_ENC_RESIDUES,
    L_NEIGHBORS,
    L_NEIGHBORS_IN,
    L_NEIGHBORS_OUT,
    L_RESIDUES,
    R_ENC_ATOMS,
    R_ENC_RESIDUES,
    R_NEIGHBORS,
    R_NEIGHBORS_IN,
    R_NEIGHBORS_OUT,
    R_RESIDUES,
    LABELS,
    N_PROTEINS,
    PROTEINS,
    TRAINING,
    VALIDATION,
    TESTING
)


class ProteinInteractionDataset(Dataset):

    """
    Dataset for protein pairs.
    """
    __metaclass__ = abc.ABCMeta
    
    PROT_NAMES = PROTEINS

    def __init__(self, dbpath, bert):
        self.env = lmdb.open(dbpath)
        self.start = PROTEINS.index(self.PROT_NAMES[0])
        self.stop = PROTEINS.index(self.PROT_NAMES[-1]) + 1
        self.size = self.stop - self.start
        self.bert = bert

    def __getitem__(self, idx):
        if (idx < 0 or idx >= self.size):
            raise IndexError()
        with self.env.begin(write=False) as txn:
            name, xdata, ydata = self._read_protein_pair(txn, idx)
        return name, xdata, ydata

    def __len__(self):
        return self.size

    def __del__(self):
        self.env.close()

    @abc.abstractmethod
    def _read_protein_pair(txn, idx):
        pass

    @staticmethod
    def _build_targets(distances):
        # The label is 1
        labels = (distances <= 7.)
        cases = labels | (distances > 8.)
        labels = labels.astype(np.float32)
        pos_weight = sum(cases) / (sum(labels) + 1)
        weights = (pos_weight - 1) * labels + cases
        targets = torch.from_numpy(labels), torch.from_numpy(weights)
        return targets
    

        

    
class AtomsDataset(ProteinInteractionDataset):

    """
    Dataset for atoms data in proteins pairs.
    """

    def _read_protein_pair(self, txn, idx):
        """
        Read atoms data of a pair of protein from the LMDB file.

        Parameters
        ----------
        txn : lmdb.Transaction
            LMDB transaction to write in.
        idx : int
            Index of the PDB file (ex: you are reading the data from PDB
            file in the environment and you are currently reading the
            idx'th one).

        Returns
        -------
        tuple of tuple of np.ndarray
            Input of the GNN.
        torch.tensor of float
            Target labels of the GNN.
        """
        prefix = f"{self.start + idx}"

        name = txn.get(prefix.encode()).decode()

        l_enc_atoms = np.frombuffer(
            txn.get(SEP.join([prefix, L_ENC_ATOMS]).encode()),
            dtype=np.float32
        )
        l_enc_residues = np.frombuffer(
            txn.get(SEP.join([prefix, L_ENC_RESIDUES]).encode()),
            dtype=np.float32
        )
        l_neighbors_in = np.frombuffer(
            txn.get(SEP.join([prefix, L_NEIGHBORS_IN]).encode()),
            dtype=np.int64
        )
        l_neighbors_out = np.frombuffer(
            txn.get(SEP.join([prefix, L_NEIGHBORS_OUT]).encode()),
            dtype=np.int64
        )
        l_residues = np.frombuffer(
            txn.get(SEP.join([prefix, L_RESIDUES]).encode()),
            dtype=np.int64
        )
        residues_feat = 1024 if self.bert else len(CATEGORIES[RESIDUES])
        xdata1 = (
            torch.from_numpy(np.copy(
                l_enc_atoms.reshape((-1, len(CATEGORIES[ATOMS])))
            )),
            torch.from_numpy(np.copy(
                l_enc_residues.reshape((-1, residues_feat))
        )),
        torch.from_numpy(np.copy(
            l_neighbors_in.reshape((-1, 10))
        )),
        torch.from_numpy(np.copy(
            l_neighbors_out.reshape((-1, 10))
        )),
        torch.from_numpy(np.copy(l_residues))
        )

        r_enc_atoms = np.frombuffer(
            txn.get(SEP.join([prefix, R_ENC_ATOMS]).encode()),
            dtype=np.float32
        )
        r_enc_residues = np.frombuffer(
            txn.get(SEP.join([prefix, R_ENC_RESIDUES]).encode()),
            dtype=np.float32
        )
        r_neighbors_in = np.frombuffer(
            txn.get(SEP.join([prefix, R_NEIGHBORS_IN]).encode()),
            dtype=np.int64
        )
        r_neighbors_out = np.frombuffer(
            txn.get(SEP.join([prefix, R_NEIGHBORS_OUT]).encode()),
            dtype=np.int64
        )
        r_residues = np.frombuffer(
            txn.get(SEP.join([prefix, R_RESIDUES]).encode()),
            dtype=np.int64
        )
        xdata2 = (
            torch.from_numpy(np.copy(
                r_enc_atoms.reshape((-1, len(CATEGORIES[ATOMS])))
            )),
            torch.from_numpy(np.copy(
                r_enc_residues.reshape((-1, residues_feat))
            )),
            torch.from_numpy(np.copy(
                r_neighbors_in.reshape((-1, 10))
            )),
        torch.from_numpy(np.copy(
            r_neighbors_out.reshape((-1, 10))
        )),
            torch.from_numpy(np.copy(r_residues))
        )
        
        distances = np.frombuffer(
            txn.get(SEP.join([prefix, LABELS]).encode()),
            dtype=np.float32
        )
        targets = self._build_targets(distances)
        return name, (xdata1, xdata2), targets




class AtomsTrainingDataset(AtomsDataset):
    """
    Specific class for the training dataset.
    """
    PROT_NAMES = TRAINING


class AtomsValidationDataset(AtomsDataset):
    """
    Specific class for the validation dataset.
    """
    PROT_NAMES = VALIDATION


class AtomsTestingDataset(AtomsDataset):
    """
    Specific class for the testing dataset.
    """
    PROT_NAMES = TESTING


class ResiduesDataset(ProteinInteractionDataset):

    """
    Dataset for atoms data in proteins pairs.
    """

    def _read_protein_pair(self, txn, idx):
        """
        Read residues data of a pair of protein from the LMDB file.
        
        Parameters
        ----------
        txn : lmdb.Transaction
            LMDB transaction to write in.
        idx : int
            Index of the PDB file (ex: you are reading the data from PDB
            file in the environment and you are currently reading the
            idx'th one).

        Returns
        -------
        tuple of tuple of np.ndarray
            Input of the GNN.
        torch.tensor of float
            Target labels of the GNN.
        """
        prefix = f"{self.start + idx}"

        name = txn.get(prefix.encode()).decode()

        l_enc_residues = np.frombuffer(
            txn.get(SEP.join([prefix, L_ENC_RESIDUES]).encode()),
            dtype=np.float32
        )
        l_neighbors = np.frombuffer(
            txn.get(SEP.join([prefix, L_NEIGHBORS]).encode()),
            dtype=np.int64
        )
        residues_feat = 1024 if self.bert else len(CATEGORIES[RESIDUES])
        xdata1 = (
            torch.from_numpy(np.copy(
                l_enc_residues.reshape((-1, residues_feat))
            )),
            torch.from_numpy(np.copy(
                l_neighbors.reshape((-1, 10))
            )),
        )
        
        r_enc_residues = np.frombuffer(
            txn.get(SEP.join([prefix, R_ENC_RESIDUES]).encode()),
            dtype=np.float32
        )
        r_neighbors = np.frombuffer(
            txn.get(SEP.join([prefix, R_NEIGHBORS]).encode()),
            dtype=np.int64
        )
        xdata2 = (
            torch.from_numpy(np.copy(
                r_enc_residues.reshape((-1, residues_feat))
            )),
            torch.from_numpy(np.copy(
                r_neighbors.reshape((-1, 10))
            )),
        )
        
        distances = np.frombuffer(
            txn.get(SEP.join([prefix, LABELS]).encode()),
            dtype=np.float32
        )
        targets = self._build_targets(distances)
        return name, (xdata1, xdata2), targets



class ResiduesTrainingDataset(ResiduesDataset):
    """
    Specific class for the training dataset.
    """
    PROT_NAMES = TRAINING


class ResiduesValidationDataset(ResiduesDataset):
    """
    Specific class for the validation dataset.
    """
    PROT_NAMES = VALIDATION


class ResiduesTestingDataset(ResiduesDataset):
    """
    Specific class for the testing dataset.
    """
    PROT_NAMES = TESTING
