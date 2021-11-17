# -*- coding: utf-8 -*-

"""
Recoprot symbols
"""

import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEP = "."
LIGAND = "ligand"
RECEPTOR = "receptor"
ENC_ATOMS = "encoded_atoms"
ENC_RESIDUES = "encoded_residues"
NEIGHBORS = "neighbors"
NEIGHBORS_IN = "neighbors_in"
NEIGHBORS_OUT = "neighbors_out"
ATOMS = "atoms"
RESIDUES = "residues"
LABELS = "labels"
N_PROTEINS = "n_proteins"

SAME_FILE = "*.pdb"
BOUND_LIGAND = "{}_l_b.pdb"
BOUND_RECEPTOR = "{}_r_b.pdb"
UNBOUND_LIGAND = "{}_l_u.pdb"
UNBOUND_RECEPTOR = "{}_r_u.pdb"

L_ENC_ATOMS = SEP.join([LIGAND, ENC_ATOMS])
L_ENC_RESIDUES = SEP.join([LIGAND, ENC_RESIDUES])
L_NEIGHBORS = SEP.join([LIGAND, NEIGHBORS])
L_NEIGHBORS_IN = SEP.join([LIGAND, NEIGHBORS_IN])
L_NEIGHBORS_OUT = SEP.join([LIGAND, NEIGHBORS_OUT])
L_RESIDUES = SEP.join([LIGAND, RESIDUES])

R_ENC_ATOMS = SEP.join([RECEPTOR, ENC_ATOMS])
R_ENC_RESIDUES = SEP.join([RECEPTOR, ENC_RESIDUES])
R_NEIGHBORS = SEP.join([RECEPTOR, NEIGHBORS])
R_NEIGHBORS_IN = SEP.join([RECEPTOR, NEIGHBORS_IN])
R_NEIGHBORS_OUT = SEP.join([RECEPTOR, NEIGHBORS_OUT])
R_RESIDUES = SEP.join([RECEPTOR, RESIDUES])


CATEGORIES = {
    ATOMS: ['1', 'C', 'CA', 'CB', 'CG', 'CH2', 'N',
            'NH2', 'O1', 'O2', 'OG','OH', 'SE'],
    RESIDUES: ['1', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
               'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
               'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
}

RESIDUES_TABLE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y"
}


TRAINING = [
    "4CPA", "3CPH", "1EZU", "1R8S", "1DE4", "1XU1", "1FAK", "1AKJ",
    "1IBR", "1BJ1", "2HMI", "1WEJ", "1XD3", "1F51", "1E6J", "7CEI",
    "1ZHH", "1MLC", "2SNI", "1GHQ", "1HE1", "1YVB", "2A5T", "1T6B",
    "2A9K", "1Z0K", "1ZHI", "1LFD", "1AY7", "1I9R", "1GRN", "1S1Q",
    "2J7P", "1US7", "2PCC", "1B6C", "1AZS", "2IDO", "1EER", "1BGX",
    "1GPW", "1SBB", "1R0R", "1Z5Y", "2FD6", "1GP2", "1JTG", "1PXV",
    "2C0L", "2CFH", "1JPS", "1HCF", "1XQS", "1CLV", "2O3B", "3SGQ",
    "1ATN", "2HLE", "1F6M", "1M10", "1OPH", "3D5S", "1QFW", "1R6Q",
    "1Y64", "BOYV", "2UUY", "1KXP", "1WDW", "1ML0", "9QFW", "1BUH",
    "1FQ1", "1PVH", "1GL1", "3BP8", "1KLU", "1TMQ", "1FCC", "1NSN",
    "1JIW", "1E6E", "1DFJ", "2I9B", "2AYO", "1ACB", "1K74", "1OFU",
    "2HQS", "2Z0E", "1RV6", "1IQD", "2OUL", "2O8V", "1EWY", "2B4J",
    "1FFW", "1IB1", "1GLA", "2OOR", "1HIA", "1E96", "2OT3", "2VDB",
    "1RLB", "1HE8", "1QA9", "2H7V", "1I2M", "1N2C", "1OC0", "1ZLI",
    "1KXQ", "1KAC", "2I25", "1KKL", "1WQ1", "1SYX", "1EAW", "1JZD",
    "2J0T", "1FSK", "1K5D", "1AK4", "1F34", "1BVN", "1JK9", "1MAH",
    "1FQJ", "1VFB", "1A2K", "1J2J", "1NCA", "1GXD", "1UDI", "1D6R",
    "2HRK", "1H9D", "1NW9"
]


VALIDATION = [
    "2OOB", "2SIC", "1FLE", "1IRA", "2MTA", "1BKD", "1OYV", "1AVX",
    "2FJU", "1MQ8", "2BTF", "2AJF", "1K4C", "1E4K", "2JEL", "1CGI",
    "1ZM4", "1JWH", "2G77", "1I4D", "1KTZ", "1AHW", "1GCQ", "1BVK",
    "1DQJ", "2OZA", "2ABZ", "2VIS", "1EFN", "1FC2", "1JMO", "1H1V",
    "1IJK", "2NZ8", "1PPE"
]

TESTING = [
    "4GAM", "3AAA", "4H03", "1EXB", "2GAF", "2GTP", "3RVW", "3SZK",
    "4IZ7", "4GXU", "3BX7", "2YVJ", "3V6Z", "1M27", "4FQI", "4G6J",
    "3BIW", "3PC8", "3HI6", "2X9A", "3HMX", "2W9E", "4G6M", "3LVK",
    "1JTD", "3H2V", "4DN4", "BP57", "3L5W", "3A4S", "CP57", "3DAW",
    "3VLB", "3K75", "2VXT", "3G6D", "3EO1", "4JCV", "4HX3", "3F1P",
    "3AAD", "3EOA", "3MXW", "3L89", "4M76", "BAAD", "4FZA", "4LW4",
    "1RKE", "3FN1", "3S9D", "3H11", "2A1A", "3P57"
]

PROTEINS = TRAINING + VALIDATION + TESTING
