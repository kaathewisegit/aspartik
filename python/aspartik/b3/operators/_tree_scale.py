from dataclasses import dataclass, field
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, TunableOperator
from ..parameters import Tree
from ._util import sample_range


@dataclass(slots=True)
class TreeScale(Operator, TunableOperator):
    """Scales the age of the entire tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale all internal nodes by a random scale which is randomly
    picked depending on `factor` and `distribution`.
    """

    tree: Tree
    """The tree to scale."""
    distribution: Distribution
    """Distribution from which the scale is sampled."""
    rng: RNG
    weight: float = 1

    _factor: float = field(init=False, default=0.75)

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng

        low, high = self._factor, 1 / self._factor
        scale = sample_range(low, high, self.distribution, rng)

        try:
            tree.scale(scale)
        except Exception:
            return Proposal.Abort()

        ratio = log(scale) * (tree.num_internals - 2)
        return Proposal.Hastings(ratio)

    def set_tuning(self, parameter: float) -> None:
        self._factor = parameter
