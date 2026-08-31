from pathlib import Path

from aspartik.data.msa import MSA
from aspartik.rng import RNG


def test_pathlike():
    class Pathlike:
        def __fspath__(self):
            return "data/alignments/apes.fasta"

    MSA.from_fasta_file(Pathlike())


def test_eq():
    alignments = Path("data/alignments/")

    for fasta_file in alignments.glob("*.fasta"):
        a = MSA.from_fasta_file(fasta_file)
        b = MSA.from_fasta_file(fasta_file)

        assert a == b

        s = open(fasta_file, "r").read()
        num_records = sum(1 for line in s.splitlines() if line.startswith(">"))

        assert a.num_sequences == num_records


def test_random(rng: RNG):
    for num_sequences in [1, 3, 10, 100]:
        for num_sites in [1, 3, 10, 100]:
            names = [str(i) for i in range(num_sequences)]
            msa = MSA.random(num_sequences, num_sites, names, rng)
            assert msa.num_sequences == num_sequences
            assert msa.num_sites == num_sites
            assert msa.sequence_names() == names


def test_reproducible():
    rng_a = RNG(4)
    msa_a = MSA.random(100, 1000, [str(i) for i in range(100)], rng_a)

    rng_b = RNG(4)
    msa_b = MSA.random(100, 1000, [str(i) for i in range(100)], rng_b)

    assert msa_a == msa_b
