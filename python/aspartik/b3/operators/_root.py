from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Tree
from ..tree import Internal, Node
from ._util import assert_factor, sample_range


@dataclass(slots=True)
class RootSlide(Operator):
    """
    Scales the height of the root node

    The height will be scaled between `factor` and `1 / factor` sampled with
    `distribution`.  If the move will put the root below one of its children
    the operation gets rejected.
    """

    tree: Tree
    """The tree to edit."""
    factor: float
    distribution: Distribution
    """The distribution to draw the height move distance from"""
    rng: RNG
    weight: float = 1

    def __post_init__(self):
        assert_factor(self)

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng
        root = tree.root

        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, rng)

        old_height = tree.height_of(root)
        new_height = old_height * scale

        left, right = tree.children_of(root)
        if new_height < max(tree.height_of(left), tree.height_of(right)):
            return Proposal.Reject()

        tree.set_height(root, new_height)

        return Proposal.Hastings(-log(scale))
