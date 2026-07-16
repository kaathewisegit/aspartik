from aspartik.data import AminoAcid


def test_members():
    assert AminoAcid.Alanine == AminoAcid("A")
    assert AminoAcid.Cysteine == AminoAcid("C")
    assert AminoAcid.AsparticAcid == AminoAcid("D")
    assert AminoAcid.GlutamicAcid == AminoAcid("E")
    assert AminoAcid.Phenylalanine == AminoAcid("F")
    assert AminoAcid.Glycine == AminoAcid("G")
    assert AminoAcid.Histidine == AminoAcid("H")
    assert AminoAcid.Isoleucine == AminoAcid("I")
    assert AminoAcid.Lysine == AminoAcid("K")
    assert AminoAcid.Leucine == AminoAcid("L")
    assert AminoAcid.Methionine == AminoAcid("M")
    assert AminoAcid.Asparagine == AminoAcid("N")
    assert AminoAcid.Proline == AminoAcid("P")
    assert AminoAcid.Glutamine == AminoAcid("Q")
    assert AminoAcid.Arginine == AminoAcid("R")
    assert AminoAcid.Serine == AminoAcid("S")
    assert AminoAcid.Threonine == AminoAcid("T")
    assert AminoAcid.Valine == AminoAcid("V")
    assert AminoAcid.Tryptophan == AminoAcid("W")
    assert AminoAcid.Tyrosine == AminoAcid("Y")


def test_lowercase():
    assert AminoAcid.Alanine == AminoAcid("a")
    assert AminoAcid.Cysteine == AminoAcid("c")
    assert AminoAcid.AsparticAcid == AminoAcid("d")
    assert AminoAcid.GlutamicAcid == AminoAcid("e")
    assert AminoAcid.Phenylalanine == AminoAcid("f")
    assert AminoAcid.Glycine == AminoAcid("g")
    assert AminoAcid.Histidine == AminoAcid("h")
    assert AminoAcid.Isoleucine == AminoAcid("i")
    assert AminoAcid.Lysine == AminoAcid("k")
    assert AminoAcid.Leucine == AminoAcid("l")
    assert AminoAcid.Methionine == AminoAcid("m")
    assert AminoAcid.Asparagine == AminoAcid("n")
    assert AminoAcid.Proline == AminoAcid("p")
    assert AminoAcid.Glutamine == AminoAcid("q")
    assert AminoAcid.Arginine == AminoAcid("r")
    assert AminoAcid.Serine == AminoAcid("s")
    assert AminoAcid.Threonine == AminoAcid("t")
    assert AminoAcid.Valine == AminoAcid("v")
    assert AminoAcid.Tryptophan == AminoAcid("w")
    assert AminoAcid.Tyrosine == AminoAcid("y")


def test_distinct():
    members = [
        AminoAcid.Alanine,
        AminoAcid.Cysteine,
        AminoAcid.AsparticAcid,
        AminoAcid.GlutamicAcid,
        AminoAcid.Phenylalanine,
        AminoAcid.Glycine,
        AminoAcid.Histidine,
        AminoAcid.Isoleucine,
        AminoAcid.Lysine,
        AminoAcid.Leucine,
        AminoAcid.Methionine,
        AminoAcid.Asparagine,
        AminoAcid.Proline,
        AminoAcid.Glutamine,
        AminoAcid.Arginine,
        AminoAcid.Serine,
        AminoAcid.Threonine,
        AminoAcid.Valine,
        AminoAcid.Tryptophan,
        AminoAcid.Tyrosine,
    ]
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            assert a != b


def test_repr():
    assert repr(AminoAcid.Alanine) == "AminoAcid.Alanine"
    assert repr(AminoAcid.AsparticAcid) == "AminoAcid.AsparticAcid"
    assert repr(AminoAcid.GlutamicAcid) == "AminoAcid.GlutamicAcid"


def test_str():
    assert str(AminoAcid.Alanine) == "Alanine"
    assert str(AminoAcid.AsparticAcid) == "Aspartic acid"
    assert str(AminoAcid.GlutamicAcid) == "Glutamic acid"
    assert str(AminoAcid.Tryptophan) == "Tryptophan"
