import math
from dataclasses import dataclass, field
from typing import Literal

from ...rng import RNG
from ...stats.distributions import Uniform
from .. import Operator, Proposal, TunableOperator
from ..parameters import Parameter, Real

UNIFORM01 = Uniform(0, 1)


@dataclass(slots=True)
class RandomWalk(Operator, TunableOperator):
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

    _tuning: float = field(init=False, default=0.75)

    def propose(self) -> Proposal:
        lower, upper = self.lower, self.upper
        rng = self.rng
        param = self.param

        diff = UNIFORM01.sample(rng) * self.window * (1 - self._tuning)
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

    def parameters(self) -> list[Parameter]:
        return [self.param]

    def set_tuning(self, parameter: float) -> None:
        self._tuning = parameter

    def get_tuning(self) -> float:
        return self._tuning
