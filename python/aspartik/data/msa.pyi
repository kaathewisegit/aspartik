from collections.abc import Sequence

from .fasta import DNARecord

class MSA:
    @classmethod
    def from_fasta(cls, fasta: Sequence[DNARecord]) -> MSA:
        """"""

    def deduplicate(self) -> None:
        """
        Remove identical sites

        For every set of sites with identical characters all but the first one
        are dropped from the MSA.
        """
