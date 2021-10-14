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

    s, ptr = _needleman_wunsch_matrix(seq1, seq2)    
    indices1, indices2 = _align(seq1, seq2, s, ptr)
    res1 = [residues1[i] for i in indices1]
    res2 = [residues2[i] for i in indices2]
    logging.info(f"        Alignment bound structure: {len(res1)} / {len(residues1)}.")
    logging.info(f"        Alignment unbound structure: {len(res2)} / {len(residues2)}.")
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
    n = len(seq1)
    m = len(seq2)
    s = np.zeros( (n+1, m+1) ) # DP matrix
    ptr = np.zeros( (n+1, m+1), dtype=int  ) # matrix of pointers

    ##### INITIALIZE SCORING MATRIX (base case) #####

    for i in range(1, n+1) :
        s[i,0] = indel * i
    for j in range(1, m+1):
        s[0,j] = indel * j
        
    ########## INITIALIZE TRACEBACK MATRIX ##########

    # Tag first row by LEFT, indicating initial '-'s
    ptr[0,1:] = LEFT
        
    # Tag first column by UP, indicating initial '-'s
    ptr[1:,0] = UP

    #####################################################

    for i in range(1,n+1):
        for j in range(1,m+1): 
            # match
            if seq1[i-1] == seq2[j-1]:
                s[i,j] = s[i-1,j-1] + match
                ptr[i,j] = DIAG
            # mismatch
            else :
                s[i,j] = s[i-1,j-1] + mismatch
                ptr[i,j] = DIAG
            # indel penalty
            if s[i-1,j] + indel > s[i,j] :
                s[i,j] = s[i-1,j] + indel
                ptr[i,j] = UP
            # indel penalty
            if s[i, j-1] + indel > s[i,j]:
                s[i,j] = s[i, j-1] + indel
                ptr[i,j] = LEFT

    return s, ptr



def _align(seq1, seq2, s, ptr):

    #### TRACE BEST PATH TO GET ALIGNMENT ####
    indices1 = []
    indices2 = []
    i, j = len(seq1), len(seq2)
    curr = ptr[i, j]

    while (i > 0 or j > 0):        
        ptr[i,j] += 3
        if curr == DIAG:
            if (seq1[i - 1] == seq2[j - 1]):
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
