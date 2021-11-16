# -*- coding: utf-8 -*-

"""
Preprocess main

Main preprocessing script.
"""

# Standard Library
import logging

# External Dependencies
import lmdb

# Recoprot
from .preprocessor import PreprocessorOptions
from .atoms import AtomsPreprocessor
from .residues import ResiduesPreprocessor
from ..symbols import PROTEINS


def preprocess_main():
    """
    Main function for setuptools entrypoint.
    """
    options = PreprocessorOptions.parse_args()
    preprocess_all(options)


def preprocess_all(options):
    """
    Full preprocessing based on user parser options.

    Parameters
    ----------
    options : Options
        Options given by the user through ArgParse.
    """
    preprocess = build_preprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            logging.info("%d/%d : Preprocessing protein %s",
                         i + 1, len(PROTEINS), pname)            
            preprocess.preprocess(pname, txn, i)
    envw.close()

def build_preprocessor(options):
    """
    Builder of the preprocessor corresponding to the data of interest.
    """
    return AtomsPreprocessor(options) if options.atoms else ResiduesPreprocessor(options)
