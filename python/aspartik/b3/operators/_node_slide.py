from dataclasses import dataclass

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Tree
from ._util import scale_on_range


@dataclass(slots=True)
class NodeSlide(Operator):
    """Slides the age of a random internal node between its parent and children

    This operator is similar to BEAST2's `EpochFlexOperator`: it will only
    affect the age of the selected node without altering the tree topology (a
    node cannot slide above its parent).
    """

    tree: Tree
    """The tree to edit."""
    distribution: Distribution
    """
    The distribution which will sample the new node height on the interval
    between its parent and the closest child.
    """
    rng: RNG
    weight: float = 1

    def propose(self) -> Proposal:
        """
        If there are no non-root internal nodes, the operator will bail with
        `Proposal.Reject`.
        """

        tree = self.tree

        # automatically fail on trees without non-root internal nodes
        if tree.num_internals == 1:
            return Proposal.Reject()

        # Pick a non-root internal node
        node = tree.random_internal(self.rng)
        parent = tree.parent_of(node)
        while parent is None:
            node = tree.random_internal(self.rng)
            parent = tree.parent_of(node)

        left, right = tree.children_of(node)

        oldest = tree.height_of(parent)
        youngest = max(tree.height_of(left), tree.height_of(right))

        (new_height, ratio) = scale_on_range(
            youngest, oldest, self.distribution, self.rng
        )

        tree.set_height(node, new_height)

        return Proposal.Hastings(ratio)
