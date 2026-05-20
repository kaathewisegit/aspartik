from typing import Protocol, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    ClassVector as ClassVector,
    Internal as Internal,
    IntVector as IntVector,
    Leaf as Leaf,
    Proposal as Proposal,
    Real as Real,
    RealVector as RealVector,
    Tree as Tree,
)

type Node = Leaf | Internal
"""Any node of the phylogenetic tree

Used for type hints in places where there isn't a need to distinguish between
internal and leaf nodes.
"""

type Parameter = ClassVector | Real | RealVector | IntVector | Tree


@runtime_checkable
class Scalable(Protocol):
    def scale(self, factor: float) -> int:
        """
        Scales all values of a parameter and returns the number of dimensions.
        """

        ...
