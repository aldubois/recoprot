# -*- coding: utf-8 -*-

"""
Tests of the preprocessor functions.
"""

import os
import warnings

import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBParser import PDBParser

from .context import recoprot


THIS_DIR = os.path.dirname(__file__)


def read_pdb_2prot_same_file(filename):
    """
    Read a PDB file and output a biopython PDBParser object.

    Parameters
    ----------
    filename: str
        Path to PDB file.

    Returns
    -------
    Tuple of two Bio.PDB.Chain.Chain
        The two proteins' chains.
    """
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("", filename)
    chains = list(structure.get_chains())
    return chains[0], chains[1]


def label_data(residues1, residues2, limit=6.):
    """
    Determines if residues from two chains interact with each other.

    Parameters
    ----------
    residues1 : list of Bio.PDB.Residue.Residue
        List of residues to consider in protein1.
    residues2 : list of Bio.PDB.Residue.Residue
        List of residues to consider in protein2.
    limit: float
        Distance limit in Angstrom.

    Returns
    -------
    np.ndarray of float
        For each pair of residue, indicate if the two
        residues interact with each other.
    """
    distances = recoprot.AtomsPreprocessor._compute_residues_alpha_carbon_distance(
        residues1,
        residues2
    )
    return (distances <= limit).astype(np.float32)


def test_reader_same_file():
    """
    Verify PDB reading file.
    """
    fname = os.path.join(THIS_DIR, "data", "same_file", "model.000.00.pdb")
    chain1, chain2 = read_pdb_2prot_same_file(fname)
    atom1 = next(chain1.get_atoms())
    atom2 = next(chain2.get_atoms())
    assert str(atom1.get_vector()) == "<Vector 9.74, 68.48, 8.62>"
    assert str(atom2.get_vector()) == "<Vector 13.96, 72.53, 7.27>"
    return    


def test_reader_diff_file():
    """
    Verify PDB reading file.
    """
    bdir = os.path.join(THIS_DIR, "data", "diff_file")
    fname1 = os.path.join(bdir, "1A2K_l_b.pdb")
    fname2 = os.path.join(bdir, "1A2K_r_b.pdb")
    chain1, chain2 = recoprot.AtomsPreprocessor._read_prot(fname1, fname2)
    atom1 = next(chain1.get_atoms())
    atom2 = next(chain2.get_atoms())
    assert str(atom1.get_vector()) == "<Vector 69.13, 20.06, 76.59>"
    assert str(atom2.get_vector()) == "<Vector 28.19, 5.02, 62.68>"
    return    


def test_encode_protein_atoms():
    """
    Test the encoding of a list of atoms compared to a manually compared one.
    """    
    atoms = ['O2', 'CB', "H", "N", "SE", "O", "CG"]
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    calc = recoprot.AtomsPreprocessor._encode_protein_atoms(atoms)
    assert (calc.toarray() == ref).all()
    return


def test_encode_protein_residues():
    """
    Test the encoding of a list of residues compared to a manually compared one.
    """
    residues = ['ILE', 'TRP', "ARG", "ZZZ", "GLN", "PHE", "ASN"]
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    calc = recoprot.AtomsPreprocessor._encode_protein_residues(residues)
    assert (calc.toarray() == ref).all()
    return


def test_encode_neighbors():
    """
    Test the encoding of the closest neighbors.

    This function result is compared to the result given by Soumyadip's original function.
    """    
    fname = os.path.join(THIS_DIR, "data", "same_file", "model.000.00.pdb")
    chain, _ = read_pdb_2prot_same_file(fname)
    atoms = list(chain.get_atoms())
    calc_in, calc_out = recoprot.AtomsPreprocessor._encode_neighbors(atoms)
    ref_in, ref_out = ref_neigh1(np.array(atoms))
    assert (ref_in == calc_in).all()
    assert (ref_out == calc_out).all()
    return


def test_label_data():
    """
    """
    fname = os.path.join(THIS_DIR, "data", "same_file", "model.000.00.pdb")
    chain1, chain2 = read_pdb_2prot_same_file(fname)
    labels = label_data(chain1, chain2)
    # Computed manually distances on a few cases
    assert labels[0] == False
    assert labels[1] == True
    assert labels[2] == True
    assert labels[87] == False
    assert labels[88] == False
    assert labels[78] == False
    return


def ref_neigh1(atom_list):
    """
    Reference function coming from Soumyadip repository.

    https://github.com/soumyadip1997/QA/blob/main/neigh.py
    """
    p4=NeighborSearch(atom_list)
    neighbour_list=p4.search_all(6,level="A")
    neighbour_list=np.array(neighbour_list)
    
    #dist is the distance between the neighbour and the source atom  i.e. dimension is N*1
    dist=np.array(neighbour_list[:,0]-neighbour_list[:,1])
    #sorting in ascending order
    place=np.argsort(dist)
    sorted_neighbour_list=neighbour_list[place]
    
    #old_atom_number is used for  storing atom id of the original protein before sorting
    #old_residue_number is used for storing residue number of the original protein before sorting
    source_vertex_list_atom_object=np.array(sorted_neighbour_list[:,0])
    len_source_vertex=len(source_vertex_list_atom_object)
    neighbour_vertex_with_respect_each_source_atom_object=np.array(sorted_neighbour_list[:,1])
    old_atom_number=[]
    old_residue_number=[]
    for i in atom_list:
        old_atom_number.append(i.get_serial_number())
        old_residue_number.append(i.get_parent().get_id()[1])
    old_atom_number=np.array(old_atom_number)
    old_residue_number=np.array(old_residue_number)
    req_no=len(neighbour_list)
    total_atoms=len(atom_list)
    #neigh_same_res is the 2D numpy array to store the indices of the  neighbours of  same residue and is of the shape N*10 where N is the total number of atoms 
    #neigh_diff_res is 2D numpy array to store  the indices of the  neighbours of different residue
    #same_flag is used to restrict the neighbours belonging to same residue  to 10
    #diff_flag is used to restrict the neighbours belonging to different residue to 10
    neigh_same_res=np.array([[-1]*10 for i in range(total_atoms)])
    neigh_diff_res=np.array([[-1]*10 for i in range(total_atoms)])
    same_flag=[0]*total_atoms
    diff_flag=[0]*total_atoms
    for i in range(len_source_vertex):
        source_atom_id=source_vertex_list_atom_object[i].get_serial_number()
        neigh_atom_id=neighbour_vertex_with_respect_each_source_atom_object[i].get_serial_number()
        source_atom_res=source_vertex_list_atom_object[i].get_parent().get_id()[1]
        neigh_atom_res=neighbour_vertex_with_respect_each_source_atom_object[i].get_parent().get_id()[1]
        #finding out index of the source and neighbouring atoms from the original atom array with respect to their residue id and atom id    
        temp_index1=np.where(source_atom_id==old_atom_number)[0]

        temp_index2=np.where(neigh_atom_id==old_atom_number)[0]
        for i1 in temp_index1:
            if old_residue_number[i1]==source_atom_res:
                source_index=i1
                break
        for i1 in temp_index2:
            if old_residue_number[i1]==neigh_atom_res:
                neigh_index=i1
                break
        #if both the residues are same        
        
        if source_atom_res==neigh_atom_res :

            #limiting the number of neighbours of same residue to 10

            if int(same_flag[source_index])< 10:
                neigh_same_res[source_index][same_flag[source_index]]=neigh_index
                same_flag[source_index]+=1
                
            if int(same_flag[neigh_index])< 10:
                neigh_same_res[neigh_index][same_flag[neigh_index]]=source_index
                same_flag[neigh_index]+=1
                
        # if both the residues are different
        elif source_atom_res!=neigh_atom_res :

            #limiting the number of neighbours of different residues to 10
            
            if int(diff_flag[source_index])< 10:
                neigh_diff_res[source_index][diff_flag[source_index]]=neigh_index
                diff_flag[source_index]+=1


            if int(diff_flag[neigh_index])< 10:

                neigh_diff_res[neigh_index][diff_flag[neigh_index]]=source_index
                diff_flag[neigh_index]+=1
    
    return neigh_same_res,neigh_diff_res
