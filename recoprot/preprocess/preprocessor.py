# -*- coding: utf-8 -*-

"""
Base preprocessor class.
"""

# Standard Library
import os
import abc
import logging
import argparse
import warnings

# External Dependencies
from Bio.PDB.PDBParser import PDBParser
from transformers import BertModel, BertTokenizer

# Recoprot
from ..symbols import (
    DEVICE,
    N_PROTEINS,
    PROTEINS,
    BOUND_LIGAND,
    BOUND_RECEPTOR,
    UNBOUND_LIGAND,
    UNBOUND_RECEPTOR,
)


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
        """
        Constructor from argument parser.
        """
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
                f" db_size={self.db_size},"
                f" proteins={self.proteins},"
                f" protbert={self.protbert})")


class Preprocessor:

    """
    Preprocessor Base Class.
    """

    __metaclass__ = abc.ABCMeta

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
        """
        Write the context in the LMDB database.
        """
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

        xdata, labels = self._preprocess_structure(
            pname,
            self.options.inp,
            self.tokenizer,
            self.model
        )
        self._verify_data(xdata, labels)
        self._write_data(pname, xdata, labels, txn, i)
        print()


    @staticmethod
    @abc.abstractmethod
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

    @staticmethod
    @abc.abstractmethod
    def _verify_data(xdata, labels):
        """
        Verify that the dimensions of the data and of the labels match.
        """

    @staticmethod
    @abc.abstractmethod
    def _write_data(name, xdata, labels, txn, idx):
        """
        Write data input of the GNN and its labels.
        """

    @staticmethod
    def _read_structure(folder, name):
        """
        Read a full protein structure.
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

        return res_l_b, res_r_b, res_l_u, res_r_u

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
