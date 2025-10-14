from dataclasses import dataclass
from math import log
from typing import SupportsFloat

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, Real
from ._util import sample_range


@dataclass(slots=True)
class UpDown(Operator):
    """
    TODO
    """

    up: SupportsFloat
    """The parameter to scale up."""
    down: SupportsFloat
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
        if not 0 < self.factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {self.factor}")

    def propose(self) -> Proposal:
        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        num_scaling_up = 0
        num_scaling_down = 0

        if isinstance(self.up, Real):
            for i in range(len(self.up)):
                self.up[i] *= scale
                num_scaling_up += 1

        if isinstance(self.down, Real):
            for i in range(len(self.down)):
                self.down[i] *= scale
                num_scaling_down += 1

        ratio = log(scale) * (num_scaling_up - num_scaling_down - 2)
        return Proposal.Hastings(ratio)
