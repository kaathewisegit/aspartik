from dataclasses import dataclass

from ._util import scale_on_range
from .. import Proposal, Tree
from ...rng import RNG
from ...stats.distributions import Distribution


@dataclass
class EpochScale:
    """Scales a random epoch in a tree.

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the weight of 0).
    """

    tree: Tree
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
        if not 0 < self.factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {self.factor}")

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng

        low, high = self.factor, 1 / self.factor
        scale, ratio = scale_on_range(low, high, self.distribution, self.rng)

        x = tree.random_internal(rng)
        y = tree.random_internal(rng)
        lower = min(tree.weight_of(x), tree.weight_of(y))
        upper = max(tree.weight_of(x), tree.weight_of(y))

        move_to = lower + scale * (upper - lower)
        delta = move_to - upper

        num_scaled = 0
        for node in tree.internals():
            if lower < tree.weight_of(node) <= upper:
                new_weight = lower + scale * (tree.weight_of(node) - lower)
                tree.update_weight(node, new_weight)
                num_scaled += 1
            elif tree.weight_of(node) > upper:
                new_weight = tree.weight_of(node) + delta
                tree.update_weight(node, new_weight)

        if num_scaled < 2:
            return Proposal.Reject()

        ratio = num_scaled * ratio
        return Proposal.Hastings(ratio)
