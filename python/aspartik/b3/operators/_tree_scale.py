from math import log
from dataclasses import dataclass

from ._util import sample_range
from .. import Proposal, Tree
from ...rng import Rng
from ...stats.distributions import Distribution


@dataclass
class TreeScale:
    """Full tree scale operator.

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the weight of 0).
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
    rng: Rng
    weight: float = 1

    def __post_init__(self):
        if not 0 < self.factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {self.factor}")

    def propose(self) -> Proposal:
        tree = self.tree

        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        for node in tree.nodes():
            new_weight = tree.weight_of(node) * scale
            tree.update_weight(node, new_weight)

        ratio = log(scale) * (tree.num_internals - 2)
        return Proposal.Hastings(ratio)
