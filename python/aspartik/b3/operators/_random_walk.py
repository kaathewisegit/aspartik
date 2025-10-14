from dataclasses import dataclass, field

from ...rng import RNG
from ...stats.distributions import Uniform
from .. import Operator, Proposal, Tree
from ..parameters import Real
from ._util import sample_range


@dataclass(slots=True)
class RandomWalk(Operator):
    """
    Adds or subtracts values to a parameter irrespective of its value

    The delta is sampled uniformly from `[0, window)` and is negated half of
    the time.

    This parameter doesn't have bounds.  Use the `Bound` prior to restrict
    parameter values.
    """

    param: Real
    window: float
    rng: RNG
    weight: float = 1

    _dist: Uniform = field(default_factory=lambda: Uniform(0, 1), init=False)

    def propose(self) -> Proposal:
        diff = sample_range(0, self.window, self._dist, self.rng)
        if self.rng.random_bool():
            diff *= -1

        # TODO: multidimensional parameters
        self.param[0] = self.param[0] + diff

        return Proposal.Hastings(0.0)
