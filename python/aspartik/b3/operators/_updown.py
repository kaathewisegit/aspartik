from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal
from ..parameters import Real, Scalable
from ._util import assert_factor, sample_range


@dataclass(slots=True)
class UpDown(Operator):
    """
    TODO
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
        except:
            return Proposal.Reject()

        ratio = log(scale) * (num_scaling_up - num_scaling_down - 2)
        return Proposal.Hastings(ratio)
