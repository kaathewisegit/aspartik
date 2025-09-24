from collections.abc import Sequence

from . import DNASeq
from .fasta import DNARecord

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
        sites than their backing buffer has.  This can happen after slicing or calling methods like `deduplicate`.
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

    def deduplicate(self) -> MSA:
        """
        Create a new MSA without repeating sites

        For every set of sites with identical characters all but the first one
        are dropped.
        """
