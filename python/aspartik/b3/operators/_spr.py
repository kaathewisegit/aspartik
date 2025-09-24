from dataclasses import dataclass

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Tree
from ._util import scale_on_range


@dataclass(slots=True)
class SubtreePruneRegraft(Operator):
    """
    TODO
    """

    tree: Tree
    """The tree to edit."""
    rng: RNG
    weight: float = 1

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng
        root = tree.root()

        node = tree.random_internal(rng)
        while node == root or tree.parent_of(node) == root:
            node = tree.random_internal(rng)

        parent = tree.parent_of(node)
        assert parent is not None  # checked in the loop

        grandparent = tree.parent_of(parent)
        assert grandparent is not None  # checked in the loop

        sibling = tree.other_child(parent, node)

        parent_height = tree.height_of(parent)

        edge = tree.random_intersecting_edge(parent_height, rng)

        if edge is None:
            return Proposal.Reject()

        other, other_parent = tree.edge_nodes(edge)

        # Tree update
        grandparent_to_parent = tree.edge_index(parent)
        parent_to_sibling = tree.edge_index(sibling)
        other_parent_to_other = tree.edge_index(other)

        tree.update_edge(grandparent_to_parent, sibling)
        tree.update_edge(other_parent_to_other, parent)
        tree.update_edge(parent_to_sibling, other)

        tree.validate()

        return Proposal.Hastings(0.0)
