from aspartik.data import DNASeq
from aspartik.data.fasta import DNARecord
from aspartik.io import FastaReader

# TODO: fuzzing, a harness, testing various reader interfaces, file vs obj


def test_one():
    r = FastaReader.from_file("python/tests/io/fasta/one.fasta")
    records = list(r)

    assert len(records) == 1
    assert records[0] == DNARecord("Record", DNASeq("AAAAAAAAAAAAAA"))


def test_two():
    r = FastaReader.from_file("python/tests/io/fasta/two.fasta")
    records = list(r)

    assert len(records) == 2
    assert records[0] == DNARecord("Record 1", DNASeq("AAAAAAAAAAAAAAAAAA"))
    assert records[1] == DNARecord("Record 2", DNASeq("CCCCCCCCCCCCCCCCCCCC"))


def test_multiline():
    r = FastaReader.from_file("python/tests/io/fasta/multiline.fasta")
    records = list(r)

    assert len(records) == 1
    assert records[0] == DNARecord("Multiline record", DNASeq("ACGTACGTTTTTTTTTTTT"))
