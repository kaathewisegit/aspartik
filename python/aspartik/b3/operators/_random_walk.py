import math
from dataclasses import dataclass, field
from typing import Literal

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
    lower: float = 0
    upper: float = math.inf
    boundary: Literal["reflect"] = "reflect"
    weight: float = 1

    _dist: Uniform = field(default_factory=lambda: Uniform(0, 1), init=False)

    def propose(self) -> Proposal:
        lower, upper = self.lower, self.upper
        rng = self.rng
        param = self.param

        diff = sample_range(0, self.window, self._dist, rng)
        if rng.random_bool():
            diff *= -1

        new_value = float(param) + diff

        if new_value < self.lower:
            match self.boundary:
                case "reflect":
                    if self.upper == math.inf:
                        new_value = lower + (lower - new_value)
                    else:
                        # TODO: ping-pong reflection
                        pass

        if new_value > self.upper:
            # TODO
            if self.lower == math.inf:
                new_value = upper + (upper - new_value)
            else:
                pass

        self.param.set(new_value)

        return Proposal.Hastings(0.0)
