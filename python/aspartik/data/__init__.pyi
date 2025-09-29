from __future__ import annotations

from typing import Self

class DNANucleotide:
    Adenine: Self
    Cytosine: Self
    Guanine: Self
    Thymine: Self

    Weak: Self
    Strong: Self
    Amino: Self
    Ketone: Self
    Purine: Self
    Pyrimidine: Self

    NotAdenine: Self
    NotCytosine: Self
    NotGuanine: Self
    NotThymine: Self

    Any: Self
    Gap: Self

    def __init__(self, char: str): ...
    def __contains__(self, other: Self) -> bool: ...
    def complement(self) -> Self: ...

class DNASeq:
    def __init__(self, sequence: str | None): ...
    def __getitem__(self, index: int) -> DNANucleotide: ...
    def __len__(self) -> int: ...
    def complement(self) -> DNASeq: ...
    def reverse_complement(self) -> DNASeq: ...

class Phred:
    def __init__(self, ch: str): ...
    def accuracy(self) -> float: ...
    def probability_incorrect(self) -> float: ...
