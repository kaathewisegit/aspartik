from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

class DNANucleotide:
    Adenine: DNANucleotide
    Cytosine: DNANucleotide
    Guanine: DNANucleotide
    Thymine: DNANucleotide

    Weak: DNANucleotide
    Strong: DNANucleotide
    Amino: DNANucleotide
    Ketone: DNANucleotide
    Purine: DNANucleotide
    Pyrimidine: DNANucleotide

    NotAdenine: DNANucleotide
    NotCytosine: DNANucleotide
    NotGuanine: DNANucleotide
    NotThymine: DNANucleotide

    Any: DNANucleotide
    Gap: DNANucleotide

    def __init__(self, char: str): ...
    def __contains__(self, other: DNANucleotide) -> bool: ...
    def complement(self) -> DNANucleotide: ...

class DNASeq:
    def __init__(self, sequence: str | None): ...
    def __getitem__(self, index: int) -> DNANucleotide: ...
    def __len__(self) -> int: ...
    def complement(self) -> DNASeq: ...
    def reverse_complement(self) -> DNASeq: ...

class DNARecord:
    def __init__(self, name: str, sequence: DNASeq): ...
    @property
    def sequence(self) -> DNASeq: ...
    @property
    def description(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __len__(self) -> int: ...

class Phred:
    def __init__(self, ch: str): ...
    def accuracy(self) -> float: ...
    def probability_incorrect(self) -> float: ...

class MSA:
    @classmethod
    def from_fasta(cls, fasta: Sequence[DNARecord]) -> MSA:
        """
        Create an alignment from FASTA records

        All records must have the same length.
        """

    @property
    def num_sites(self) -> int:
        """
        Number of sites in the alignment

        Some alignments might be partial, that is, they might reference fewer
        sites than their backing buffer has.  This can happen after slicing or
        some methods.
        """

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the alignment"""

    def sequence_name(self, index: int) -> str:
        """The name (or id) of the `index`'th sequence"""

    def sequence_names(self) -> list[str]:
        """A list of all sequence names in the alignment"""

    def sequence(self, index: int) -> DNASeq:
        """The `index`'th sequence in the alignment"""

    def base_frequencies(self) -> tuple[float, float, float, float]:
        """
        Base frequencies of Adenine, Cytosine, Guanine, and Thymine, in that
        order
        """

    from typing import Optional

class Tree:
    def __init__(self, newick: Optional[str] = None): ...
    def __str__(self) -> str: ...
