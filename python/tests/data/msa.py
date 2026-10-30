from pathlib import Path

from aspartik.data import DNASeq, TaxonSet
from aspartik.data.fasta import DNARecord
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


def test_from_records():
    records = [
        DNARecord(">a", DNASeq("AC")),
        DNARecord(">b", DNASeq("GT")),
    ]

    msa = MSA.from_records(records)

    assert msa.num_sequences == 2
    assert msa.num_sites == 2
    assert msa.sequence_names() == TaxonSet(["a", "b"])
    assert str(msa.sequence(0)) == "AC"
    assert str(msa.sequence(1)) == "GT"


def test_random(rng: RNG):
    for num_sequences in [1, 3, 10, 100]:
        names = TaxonSet.ranged_ints(num_sequences)
        for num_sites in [1, 3, 10, 100]:
            msa = MSA.random(num_sequences, num_sites, names, rng)
            assert msa.num_sequences == num_sequences
            assert msa.num_sites == num_sites
            assert msa.sequence_names() == names


def test_reproducible():
    rng_a = RNG(4)
    msa_a = MSA.random(100, 1000, TaxonSet.ranged_ints(100), rng_a)

    rng_b = RNG(4)
    msa_b = MSA.random(100, 1000, TaxonSet.ranged_ints(100), rng_b)

    assert msa_a == msa_b
