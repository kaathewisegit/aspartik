from dataclasses import dataclass, field
from math import log

from ...rng import RNG
from ...stats.distributions import Distribution
from .. import Operator, Proposal, TunableOperator
from ..parameters import Real
from ._util import sample_range


@dataclass(slots=True)
class ParamScale(Operator, TunableOperator):
    """Scales one parameter

    This operator is analogous to BEAST2's `ScaleOperator`, except it only
    works for parameters.

    Note that this operator doesn't have the upper/lower bounds BEAST2's analog
    has.  Instead the `Bound` prior should be used to put limits on the
    parameter values.
    """

    param: Real
    """The parameter to scale."""
    distribution: Distribution
    """The distribution from which to sample the scaling factor."""
    rng: RNG
    weight: float = 1

    _factor: float = field(init=False, default=0.75)

    def propose(self) -> Proposal:
        low, high = self._factor, 1 / self._factor
        scale = sample_range(low, high, self.distribution, self.rng)

        self.param.scale(scale)

        ratio = -log(scale)
        return Proposal.Hastings(ratio)

    def set_tuning(self, parameter: float) -> None:
        self._factor = parameter

    def get_tuning(self) -> float:
        return self._factor
