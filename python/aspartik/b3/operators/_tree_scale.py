from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Internal, Node, Operator, Proposal, Tree
from ._util import assert_factor, sample_range


@dataclass(slots=True)
class TreeScale(Operator):
    """Scales the age of the entire tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale all internal nodes by a random scale which is randomly
    picked depending on `factor` and `distribution`.
    """

    tree: Tree
    """The tree to scale."""
    factor: float
    """
    The scaling ratio will be sampled from `(factor, 1 / factor)`.  So, the
    factor must be between 0 and 1 and the smaller it is the larger the steps
    will be.
    """
    distribution: Distribution
    """Distribution from which the scale is sampled."""
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

        try:
            tree.scale(scale)
        except:
            return Proposal.Reject()

        ratio = log(scale) * (tree.num_internals - 2)
        return Proposal.Hastings(ratio)
