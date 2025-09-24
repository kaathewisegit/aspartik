from ..data.msa import MSA
from . import FastaReader


def read_msa_from_fasta(path: str) -> MSA:
    records = list(FastaReader.from_file(path))
    msa = MSA.from_fasta(records)
    return msa
