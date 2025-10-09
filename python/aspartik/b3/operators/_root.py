from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Tree
from ..tree import Internal, Node
from ._util import scale_on_range


@dataclass(slots=True)
class RootSlide(Operator):
    """
    TODO
    """

    tree: Tree
    """The tree to edit."""
    factor: float
    distribution: Distribution
    """The distribution to draw the height move distance from"""
    rng: RNG
    weight: float = 1

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng
        root = tree.root

        low, high = self.factor, 1 / self.factor
        new_height, ratio = scale_on_range(low, high, self.distribution, rng)

        left, right = tree.children_of(root)
        if new_height < max(tree.height_of(left), tree.height_of(right)):
            return Proposal.Reject()

        tree.set_height(root, new_height)

        return Proposal.Hastings(ratio)
