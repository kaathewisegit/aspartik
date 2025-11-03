from aspartik.data import DNANucleotide


def test_members():
    assert DNANucleotide.Adenine == DNANucleotide("A")
    assert DNANucleotide.Cytosine == DNANucleotide("C")
    assert DNANucleotide.Guanine == DNANucleotide("G")
    assert DNANucleotide.Thymine == DNANucleotide("T")

    assert DNANucleotide.Weak == DNANucleotide("W")
    assert DNANucleotide.Strong == DNANucleotide("S")
    assert DNANucleotide.Amino == DNANucleotide("M")
    assert DNANucleotide.Ketone == DNANucleotide("K")
    assert DNANucleotide.Purine == DNANucleotide("R")
    assert DNANucleotide.Pyrimidine == DNANucleotide("Y")

    assert DNANucleotide.NotAdenine == DNANucleotide("B")
    assert DNANucleotide.NotCytosine == DNANucleotide("D")
    assert DNANucleotide.NotGuanine == DNANucleotide("H")
    assert DNANucleotide.NotThymine == DNANucleotide("V")

    assert DNANucleotide.Any == DNANucleotide("N")
    assert DNANucleotide.Gap == DNANucleotide("-")


def test_complement():
    assert DNANucleotide.Adenine.complement() == DNANucleotide.Thymine
    assert DNANucleotide.Cytosine.complement() == DNANucleotide.Guanine
    assert DNANucleotide.Guanine.complement() == DNANucleotide.Cytosine
    assert DNANucleotide.Thymine.complement() == DNANucleotide.Adenine

    assert DNANucleotide.Weak.complement() == DNANucleotide.Strong
    assert DNANucleotide.Strong.complement() == DNANucleotide.Weak
    assert DNANucleotide.Amino.complement() == DNANucleotide.Ketone
    assert DNANucleotide.Ketone.complement() == DNANucleotide.Amino
    assert DNANucleotide.Purine.complement() == DNANucleotide.Pyrimidine
    assert DNANucleotide.Pyrimidine.complement() == DNANucleotide.Purine

    assert DNANucleotide.NotAdenine.complement() == DNANucleotide.NotThymine
    assert DNANucleotide.NotCytosine.complement() == DNANucleotide.NotGuanine
    assert DNANucleotide.NotGuanine.complement() == DNANucleotide.NotCytosine
    assert DNANucleotide.NotThymine.complement() == DNANucleotide.NotAdenine

    assert DNANucleotide.Any.complement() == DNANucleotide.Any
    assert DNANucleotide.Gap.complement() == DNANucleotide.Gap


def test_contains():
    assert DNANucleotide.Adenine in DNANucleotide.Adenine
    assert DNANucleotide.NotAdenine not in DNANucleotide.Adenine
    assert DNANucleotide.NotThymine not in DNANucleotide.Adenine

    assert DNANucleotide.Cytosine in DNANucleotide.Strong
    assert DNANucleotide.Guanine in DNANucleotide.Strong
    assert DNANucleotide.Adenine not in DNANucleotide.Strong
    assert DNANucleotide.Thymine not in DNANucleotide.Strong

    assert DNANucleotide.Strong in DNANucleotide.NotThymine
    assert DNANucleotide.Ketone in DNANucleotide.NotThymine
    assert DNANucleotide.Pyrimidine in DNANucleotide.NotThymine
