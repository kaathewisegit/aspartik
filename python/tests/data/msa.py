from aspartik.io.msa import read_msa_from_fasta


def test_eq():
    for name in ["apes", "influenza", "respiratory"]:
        a = read_msa_from_fasta(f"data/alignments/{name}.fasta")
        b = read_msa_from_fasta(f"data/alignments/{name}.fasta")

        assert a == b
