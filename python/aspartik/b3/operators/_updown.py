from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal
from ..parameters import Scalable
from ._util import assert_factor, sample_range


@dataclass(slots=True)
class UpDown(Operator):
    """
    Scales `up` by factor and `down` by inverse factor on each step

    See `factor` documentation for how the scaling factor is sampled.  Note
    that it can be both more and less than 1, so `up` can sometimes be scaled
    down (and so `down` would be scaled up on those steps).

    Though, when using the uniform distribution, scaling factor will be biased
    towards values `> 1`, so `up` will be scaled up more often than not (not
    accounting for rejections).
    """

    up: Scalable
    """The parameter to scale up."""
    down: Scalable
    """The parameter to scale down."""
    factor: float
    """
    The scale ratio will be sampled from `(factor, 1 / factor)`.  So, the
    smaller the factor, the larger the moves proposed by this operator are.
    This also means that `factor` must be within `(0, 1)`.
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

        try:
            num_scaling_up = self.up.scale(scale)
            num_scaling_down = self.down.scale(1 / scale)
        except Exception:
            return Proposal.Abort()

        ratio = log(scale) * (num_scaling_up - num_scaling_down - 2)
        return Proposal.Hastings(ratio)
