import os
import pickle

from aspartik.data.msa import MSA
from aspartik.io.msa import read_msa_from_fasta


def test_pickle_roundtrip():
    for name in ["apes", "influenza", "respiratory"]:
        old_msa = read_msa_from_fasta(f"data/alignments/{name}.fasta")
        new_msa = pickle.loads(pickle.dumps(old_msa))

        assert new_msa == old_msa
