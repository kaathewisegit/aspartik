from dataclasses import dataclass, field
from math import log

from ...distributions import Sample
from ...rng import RNG
from .. import Operator, Proposal, TunableOperator
from ..parameters import Parameter, Real, Tree
from ._util import sample_range


@dataclass(slots=True)
class UpDown(Operator, TunableOperator):
    """
    Scales `up` by factor and `down` by inverse factor on each step

    See `factor` documentation for how the scaling factor is sampled.  Note
    that it can be both more and less than 1, so `up` can sometimes be scaled
    down (and so `down` would be scaled up on those steps).

    Though, when using the uniform distribution, scaling factor will be biased
    towards values `> 1`, so `up` will be scaled up more often than not (not
    accounting for rejections).
    """

    up: Real | Tree
    """The parameter to scale up."""
    down: Real | Tree
    """The parameter to scale down."""
    distribution: Sample[float]
    """The distribution from which to sample the scaling factor."""
    rng: RNG
    weight: float = 1

    _factor: float = field(init=False, default=0.75)

    def propose(self) -> Proposal:
        low, high = self._factor, 1 / self._factor
        scale = sample_range(low, high, self.distribution, self.rng)

        try:
            num_scaling_up = self.up.scale(scale)
            num_scaling_down = self.down.scale(1 / scale)
        except Exception:
            return Proposal.Abort()

        ratio = log(scale) * (num_scaling_up - num_scaling_down - 2)
        return Proposal.Hastings(ratio)

    def parameters(self) -> list[Parameter]:
        return [self.up, self.down]

    def set_tuning(self, parameter: float) -> None:
        self._factor = parameter

    def get_tuning(self) -> float:
        return self._factor
