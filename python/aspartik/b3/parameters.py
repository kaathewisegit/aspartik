from dataclasses import dataclass
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


@dataclass(slots=True)
class Root:
    tree: Tree

    def __float__(self):
        return self.tree.height_of(self.tree.root)

    def is_changed(self):
        return self.tree.is_changed()


@dataclass(slots=True)
class MRCA:
    tree: Tree
    nodes: list[Node]

    def __post_init__(self):
        assert len(self.nodes) >= 2

    def mrca(self) -> Internal:
        tree = self.tree
        nodes = self.nodes
        out = tree.mrca(nodes[0], nodes[1])
        for node in nodes[2:]:
            out = tree.mrca(out, node)
        return out

    def __float__(self):
        return self.tree.height_of(self.mrca())

    def is_changed(self):
        return self.tree.is_changed()
