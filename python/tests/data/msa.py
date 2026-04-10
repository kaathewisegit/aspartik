from pathlib import Path

from aspartik.io import read_msa_from_fasta


def test_eq():
    alignments = Path("data/alignments/")

    for fasta_file in alignments.glob("*.fasta"):
        # TODO: accept paths and pathlike objects
        a = read_msa_from_fasta(str(fasta_file))
        b = read_msa_from_fasta(str(fasta_file))

        assert a == b
