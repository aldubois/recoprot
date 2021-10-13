# -*- coding: utf-8 -*-

"""
Tests of the preprocessor functions.
"""

import os
import warnings

import numpy as np
from Bio.PDB.PDBParser import PDBParser

from .context import recoprot

THIS_DIR = os.path.dirname(__file__)


def test_alignment():
    """
    Verifying alignment function compared to manual result.
    """
    fname_bound = os.path.join(THIS_DIR, "data", "diff_file", "1A2K_l_b.pdb")
    fname_unbound = os.path.join(THIS_DIR, "data", "diff_file", "1A2K_l_u.pdb")
    parser = PDBParser()
    with warnings.catch_warnings(record=True) as w:
        struct_bound = parser.get_structure("", fname_bound)
        struct_unbound = parser.get_structure("", fname_unbound)
    bound = list(next(struct_bound.get_chains()).get_residues())
    unbound = list(next(struct_unbound.get_chains()).get_residues())
    bound_res = [i.get_resname() for i in bound]
    unbound_res = [i.get_resname() for i in unbound]

    ref_bound = bound_res[:64] + bound_res[65:199] + bound_res[200:]
    ref_unbound = unbound_res[2:66] + unbound_res[67:198] + unbound_res[202:]
    assert ref_bound == ref_unbound
    
    res1, res2 = recoprot.align_proteins_residues(bound, unbound)
    calc_bound = [i.get_resname() for i in res1]
    calc_unbound = [i.get_resname() for i in res2]
    assert calc_bound == calc_unbound

    assert ref_bound == calc_unbound
    
    return
