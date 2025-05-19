from dataclasses import dataclass

from ._util import scale_on_range
from .. import Proposal, Tree
from ...rng import Rng
from ...stats.distributions import Distribution


@dataclass
class NodeSlide:
    """Operator which slides a random node between its parent and children.

    This operator is similar to BEAST2's `EpochFlexOperator`: it will only
    affect the age of the selected node without altering the tree topology (a
    node cannot slide past its parent).
    """

    tree: Tree
    """The tree to edit."""
    distribution: Distribution
    """
    The distribution which will sample the new node height on the interval
    between its parent and the closest child.
    """
    rng: Rng
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

        oldest = tree.weight_of(parent)
        youngest = max(tree.weight_of(left), tree.weight_of(right))

        (new_weight, ratio) = scale_on_range(
            youngest, oldest, self.distribution, self.rng
        )

        tree.update_weight(node, new_weight)

        return Proposal.Hastings(ratio)
