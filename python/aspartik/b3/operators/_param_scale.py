from dataclasses import dataclass
from math import log
from typing import Literal

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal
from ..parameters import Real
from ._util import assert_factor, sample_range


@dataclass(slots=True)
class ParamScale(Operator):
    """Scales one parameter

    This operator is analogous to BEAST2's `ScaleOperator`, except it only
    works for parameters.

    Note that this operator doesn't have the upper/lower bounds BEAST2's analog
    has.  Instead the `Bound` prior should be used to put limits on the
    parameter values.
    """

    param: Real
    """The parameter to scale."""
    factor: float
    """
    The scale ratio will be sampled from `(factor, 1 / factor)`.  So, the
    smaller the factor, the larger the moves proposed by this operator are.
    Also, this means that `factor` must be within `(0, 1)`.
    """
    distribution: Distribution
    """The distribution from which to sample the scaling factor."""
    rng: RNG
    weight: float = 1

    def __post_init__(self):
        assert_factor(self)

    def propose(self) -> Proposal:
        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        self.param.scale(scale)

        ratio = -log(scale)
        return Proposal.Hastings(ratio)
