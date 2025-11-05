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
    # Weak
    assert DNANucleotide.Adenine in DNANucleotide.Weak
    assert DNANucleotide.Cytosine not in DNANucleotide.Weak
    assert DNANucleotide.Guanine not in DNANucleotide.Weak
    assert DNANucleotide.Thymine in DNANucleotide.Weak

    # Strong
    assert DNANucleotide.Adenine not in DNANucleotide.Strong
    assert DNANucleotide.Cytosine in DNANucleotide.Strong
    assert DNANucleotide.Guanine in DNANucleotide.Strong
    assert DNANucleotide.Thymine not in DNANucleotide.Strong

    # Amino
    assert DNANucleotide.Adenine in DNANucleotide.Amino
    assert DNANucleotide.Cytosine in DNANucleotide.Amino
    assert DNANucleotide.Guanine not in DNANucleotide.Amino
    assert DNANucleotide.Thymine not in DNANucleotide.Amino

    # Ketone
    assert DNANucleotide.Adenine not in DNANucleotide.Ketone
    assert DNANucleotide.Cytosine not in DNANucleotide.Ketone
    assert DNANucleotide.Guanine in DNANucleotide.Ketone
    assert DNANucleotide.Thymine in DNANucleotide.Ketone

    # Purine
    assert DNANucleotide.Adenine in DNANucleotide.Purine
    assert DNANucleotide.Cytosine not in DNANucleotide.Purine
    assert DNANucleotide.Guanine in DNANucleotide.Purine
    assert DNANucleotide.Thymine not in DNANucleotide.Purine

    # Pyrimidine
    assert DNANucleotide.Adenine not in DNANucleotide.Pyrimidine
    assert DNANucleotide.Cytosine in DNANucleotide.Pyrimidine
    assert DNANucleotide.Guanine not in DNANucleotide.Pyrimidine
    assert DNANucleotide.Thymine in DNANucleotide.Pyrimidine

    # Not adenine
    assert DNANucleotide.Adenine not in DNANucleotide.NotAdenine
    assert DNANucleotide.Cytosine in DNANucleotide.NotAdenine
    assert DNANucleotide.Guanine in DNANucleotide.NotAdenine
    assert DNANucleotide.Thymine in DNANucleotide.NotAdenine

    assert DNANucleotide.Weak not in DNANucleotide.NotAdenine
    assert DNANucleotide.Strong in DNANucleotide.NotAdenine
    assert DNANucleotide.Amino not in DNANucleotide.NotAdenine
    assert DNANucleotide.Ketone in DNANucleotide.NotAdenine
    assert DNANucleotide.Purine not in DNANucleotide.NotAdenine
    assert DNANucleotide.Pyrimidine in DNANucleotide.NotAdenine
