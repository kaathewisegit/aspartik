from pathlib import Path

from aspartik.io import read_msa_from_fasta


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
