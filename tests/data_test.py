# -*- coding: utf-8 -*-

"""
Test the reading an writing to the LMDB.
"""

import lmdb
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from .context import recoprot



DATA_DIR = "tests/data/diff_file"
PROT_NAME = "1A2K"

class OneProteinAtomsDataset(recoprot.AtomsDataset):
    PROT_NAMES = [PROT_NAME]
    

class OneProteinResiduesDataset(recoprot.ResiduesDataset):
    PROT_NAMES = [PROT_NAME]
    

def test_read_write_atoms():

    """
    Test read/write of the LMDB.
    """
    
    x_ref, labels_ref = recoprot.AtomsPreprocessor._preprocess_structure(PROT_NAME, DATA_DIR)

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], False, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinAtomsDataset("/tmp/test", False)
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()
    assert (x_ref[0][2] == x_calc[0][2].numpy()).all()
    assert (x_ref[0][3] == x_calc[0][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[0][4])) == x_calc[0][4]).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()
    assert (x_ref[1][2] == x_calc[1][2].numpy()).all()
    assert (x_ref[1][3] == x_calc[1][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[1][4])) == x_calc[1][4]).all()

    assert ((labels_ref.alpha <= 6.) == labels_calc[0].numpy()).all()
    return


def test_read_write_atoms_min():

    """
    Test read/write of the LMDB.
    """
    
    x_ref, labels_ref = recoprot.AtomsPreprocessor._preprocess_structure(PROT_NAME, DATA_DIR)

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 20000000, [PROT_NAME], False, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinAtomsDataset("/tmp/test", False, False)
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()
    assert (x_ref[0][2] == x_calc[0][2].numpy()).all()
    assert (x_ref[0][3] == x_calc[0][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[0][4])) == x_calc[0][4]).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()
    assert (x_ref[1][2] == x_calc[1][2].numpy()).all()
    assert (x_ref[1][3] == x_calc[1][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[1][4])) == x_calc[1][4]).all()

    assert ((labels_ref.min <= 6.) == labels_calc[0].numpy()).all()
    return


def test_read_write_atoms_bert():

    """
    Test read/write of the LMDB.
    """
    torch.cuda.empty_cache()
    
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.to(recoprot.DEVICE)
    model = model.eval()
    x_ref, labels_ref = recoprot.AtomsPreprocessor._preprocess_structure(
        PROT_NAME, DATA_DIR, tokenizer, model
    )
    torch.cuda.empty_cache()

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test', 200000000, [PROT_NAME], True, True)
    preprocess = recoprot.AtomsPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    torch.cuda.empty_cache()

    loader = OneProteinAtomsDataset("/tmp/test", True)
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()
    assert (x_ref[0][2] == x_calc[0][2].numpy()).all()
    assert (x_ref[0][3] == x_calc[0][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[0][4])) == x_calc[0][4]).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()
    assert (x_ref[1][2] == x_calc[1][2].numpy()).all()
    assert (x_ref[1][3] == x_calc[1][3].numpy()).all()
    assert (torch.from_numpy(np.copy(x_ref[1][4])) == x_calc[1][4]).all()

    assert ((labels_ref.alpha <= 6.) == labels_calc[0].numpy()).all()
    return


def test_read_write_residues():

    """
    Test read/write of the LMDB.
    """
    
    x_ref, labels_ref = recoprot.ResiduesPreprocessor._preprocess_structure(PROT_NAME, DATA_DIR)

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test2', 20000000, [PROT_NAME], False, False)
    preprocess = recoprot.ResiduesPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinResiduesDataset("/tmp/test2", False)
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()

    assert ((labels_ref.alpha <= 6.) == labels_calc[0].numpy()).all()
    return


def test_read_write_residues_bert():

    """
    Test read/write of the LMDB.
    """
    
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.to(recoprot.DEVICE)
    model = model.eval()
    x_ref, labels_ref = recoprot.ResiduesPreprocessor._preprocess_structure(
        PROT_NAME, DATA_DIR, tokenizer, model
    )

    options = recoprot.PreprocessorOptions(DATA_DIR, '/tmp/test2', 200000000, [PROT_NAME], True, False)
    preprocess = recoprot.ResiduesPreprocessor(options)
    envw = lmdb.open(options.out, map_size=options.db_size)
    with envw.begin(write=True) as txn:
        preprocess.write_context(txn)
        for i, pname in enumerate(options.proteins):
            preprocess.preprocess(pname, txn, recoprot.PROTEINS.index(PROT_NAME))
    envw.close()

    loader = OneProteinResiduesDataset("/tmp/test2", True)
    assert(len(loader) == 1)
    name_calc, x_calc, labels_calc = loader[0]

    assert name_calc == PROT_NAME
    assert (x_ref[0][0] == x_calc[0][0].numpy()).all()
    assert (x_ref[0][1] == x_calc[0][1].numpy()).all()

    assert (x_ref[1][0] == x_calc[1][0].numpy()).all()
    assert (x_ref[1][1] == x_calc[1][1].numpy()).all()

    assert ((labels_ref.alpha <= 6.) == labels_calc[0].numpy()).all()
    return

