from pathlib import Path

from aspartik.data.msa import MSA
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def test_pathlike():
    class Pathlike:
        def __fspath__(self):
            return "data/alignments/apes.fasta"

    read_msa_from_fasta(Pathlike())


def test_eq():
    alignments = Path("data/alignments/")

    for fasta_file in alignments.glob("*.fasta"):
        a = read_msa_from_fasta(fasta_file)
        b = read_msa_from_fasta(fasta_file)

        assert a == b


def test_random(rng: RNG):
    for num_sequences in [1, 3, 10, 100]:
        for num_sites in [1, 3, 10, 100]:
            names = [str(i) for i in range(num_sequences)]
            msa = MSA.random(num_sequences, num_sites, names, rng)
            assert msa.num_sequences == num_sequences
            assert msa.num_sites == num_sites
            assert msa.sequence_names() == names
