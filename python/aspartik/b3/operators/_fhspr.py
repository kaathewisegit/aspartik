from dataclasses import dataclass

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal
from ..parameters import Tree
from ._util import assert_two_internals, family


@dataclass(slots=True)
class FixedHeightSubtreePruneRegraft(Operator):
    """
    Fixed height subtree and regraft move.

    This operator was described in [Hoehna et al 2008][l], section 3.2.7.  The
    move selects a random node `i` and its parent `i_parent`.  It then selects
    a random edge whose height overlaps with the height of `i_parent`.
    `i_parent` is spliced into the middle of this edge.

    [l]: https://alexeidrummond.org/assets/publications/2008-hoehna-evalution.pdf
    """

    tree: Tree
    """The tree to edit."""
    rng: RNG
    weight: float = 1

    def __post_init__(self):
        assert_two_internals(self)

    def propose(self) -> Proposal:
        rng = self.rng
        tree = self.tree
        root = tree.root

        node = tree.random_node(rng)
        parent = tree.parent_of(node)
        while node == root or parent == root:
            node = tree.random_node(rng)
            parent = tree.parent_of(node)

        assert parent is not None
        grandparent = tree.parent_of(parent)
        assert grandparent is not None

        sibling = tree.other_child(parent, node)

        parent_height = tree.height_of(parent)

        edge = tree.random_intersecting_edge(parent_height, rng)

        if edge is None:
            return Proposal.Reject()

        other, other_parent = tree.edge_nodes(edge)

        # regraft parent of `node` to edge between `other` its parent
        tree.spr(node, other)

        return Proposal.Hastings(0.0)
