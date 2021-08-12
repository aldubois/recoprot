# -*- coding: utf-8 -*-

"""
Utility module related to  PDB files parsing.
"""

import warnings

from biopandas.pdb import PandasPdb


def read_pdb(filename):
    """
    Read a PDB file and output a biopandas PandasPdb object.

    Parameters
    ----------
    filename: str
        Path to PDB file.

    Returns
    -------
    PandasPdb
        Object representing the PDB file.
    """
    return PandasPdb().read_pdb(filename)
