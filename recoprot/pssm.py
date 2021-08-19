# -*- coding: utf-8 -*-


"""
Windowed Position Specific Scoring Matrix.

Wrapper function to call PSI-BLAST using the BioPython package.
"""

from Bio.Blast import NCBIWWW


def call_psiblast(sequence):
    """
    Call the Psi-Blast algorithm on the FASTA sequence.

    This function call needs an internet connexion as it sends
    a call request to https://blast.ncbi.nlm.nih.gov/Blast.cgi.

    Parameters
    ----------
    sequence : str
        FASTA sequence to call psi-blast from.

    Returns
    -------
    
    """
    result = NCBIWWW.qblast(
        program="blastp",
        database="nr",
        sequence=sequence,
        service="psi"
    )
    return result
