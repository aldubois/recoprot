# -*- coding: utf-8 -*-

"""
Residues alignment functions.
"""

import logging

import numpy as np

DIAG = 0
UP = 1
LEFT = 2


def align_proteins_residues(residues1, residues2):
    """
    Align the indexes of the residues of a protein in its bound and
    unbound forms.

    Parameters
    ----------
    residues1 : list of Bio.PDB.Residue.Residue
        List of residues in protein 1.
    residues2 : list of Bio.PDB.Residue.Residue
        List of residues in protein 2.

    Returns
    -------
    list of Bio.PDB.Residue.Residue
        List of residues aligned in protein 1.
    list of Bio.PDB.Residue.Residue
        List of indexes aligned in protein 2.
    """
    seq1 = [i.get_resname() for i in residues1]
    seq2 = [i.get_resname() for i in residues2]

    ptr = _needleman_wunsch_matrix(seq1, seq2)
    indices1, indices2 = _align(seq1, seq2, ptr)
    res1 = [residues1[i] for i in indices1]
    res2 = [residues2[i] for i in indices2]
    logging.info("        Alignment bound structure: %d / %d.", len(res1), len(residues1))
    logging.info("        Alignment unbound structure: %d / %d.", len(res2), len(residues2))
    assert len(res1) == len(res2)
    return res1, res2


def _needleman_wunsch_matrix(seq1, seq2, match=1, mismatch=-1, indel=-1):
    """
    Fill the DP matrix according to the Needleman-Wunsch
    algorithm for two sequences seq1 and seq2.
    match:  the match score
    mismatch:  the mismatch score
    indel:  the indel score

    Returns the matrix of scores and the matrix of pointers
    """
    size1 = len(seq1)
    size2 = len(seq2)
    matrix = np.zeros((size1 + 1, size2 + 1)) # DP matrix
    ptr = np.zeros((size1 + 1, size2 + 1), dtype=int) # matrix of pointers

    ##### INITIALIZE SCORING MATRIX (base case) #####

    for i in range(1, size1 + 1) :
        matrix[i,0] = indel * i
    for j in range(1, size2 + 1):
        matrix[0,j] = indel * j

    ########## INITIALIZE TRACEBACK MATRIX ##########

    # Tag first row by LEFT, indicating initial '-'s
    ptr[0,1:] = LEFT

    # Tag first column by UP, indicating initial '-'s
    ptr[1:,0] = UP

    #####################################################

    for i in range(1, size1 + 1):
        for j in range(1, size2 + 1):
            # match
            if seq1[i-1] == seq2[j-1]:
                matrix[i,j] = matrix[i-1,j-1] + match
                ptr[i,j] = DIAG
            # mismatch
            else :
                matrix[i,j] = matrix[i-1,j-1] + mismatch
                ptr[i,j] = DIAG
            # indel penalty
            if matrix[i-1,j] + indel > matrix[i,j] :
                matrix[i,j] = matrix[i-1,j] + indel
                ptr[i,j] = UP
            # indel penalty
            if matrix[i, j-1] + indel > matrix[i,j]:
                matrix[i,j] = matrix[i, j-1] + indel
                ptr[i,j] = LEFT

    return ptr



def _align(seq1, seq2, ptr):

    #### TRACE BEST PATH TO GET ALIGNMENT ####
    indices1 = []
    indices2 = []
    i, j = len(seq1), len(seq2)
    curr = ptr[i, j]

    while (i > 0 or j > 0):
        ptr[i,j] += 3
        if curr == DIAG:
            if seq1[i - 1] == seq2[j - 1]:
                indices1.insert(0, i - 1)
                indices2.insert(0, j - 1)
            i -= 1
            j -= 1
        elif curr == LEFT:
            j -= 1
        elif curr == UP:
            i -= 1

        curr = ptr[i,j]

    return indices1, indices2
