from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    Internal as Internal,
    Leaf as Leaf,
    Proposal as Proposal,
    Real as Real,
    RealVector as RealVector,
    Tree as Tree,
)
from . import Stateful

type Node = Leaf | Internal
"""Any node of the phylogenetic tree

Used for type hints in places where there isn't a need to distinguish between
internal and leaf nodes.
"""

type Parameter = Real | RealVector | Tree


@runtime_checkable
class Scalable(Protocol):
    def scale(self, factor: float) -> int:
        """
        Scales all values of a parameter and returns the number of dimensions.
        """

        ...


@dataclass
class Internals(Scalable):
    tree: Tree

    def scale(self, factor: float) -> int:
        self.tree.scale(factor)

        return self.tree.num_internals


@dataclass
class Root:
    tree: Tree

    def __float__(self) -> float:
        return self.tree.height_of(self.tree.root)
