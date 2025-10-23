from dataclasses import dataclass

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Tree
from ._util import assert_two_internals, sample_range


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

    def __post_init__(self):
        assert_two_internals(self)

    def propose(self) -> Proposal:
        tree = self.tree

        node, parent = tree.random_nonroot_internal(self.rng)
        left, right = tree.children_of(node)

        oldest = tree.height_of(parent)
        youngest = max(tree.height_of(left), tree.height_of(right))

        new_height = sample_range(youngest, oldest, self.distribution, self.rng)

        tree.set_height(node, new_height)

        return Proposal.Hastings(0.0)
