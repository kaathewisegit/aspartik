from dataclasses import dataclass, field

from ...rng import RNG
from .. import Operator, Proposal, TunableOperator
from ..parameters import IntVector, Parameter, RealVector


@dataclass(slots=True)
class DeltaExchange(Operator, TunableOperator):
    """Scales a multidimensional parameter without changing its sum

    This operator is analogous to BEAST's `DeltaExchangeOperator`.  It picks
    two random dimensions from a parameter, samples a random delta from
    `distribution`, and increments one of them by delta and decrements the
    other one.
    """

    param: RealVector
    """
    The multidimensional parameter to edit.  Two random dimensions ones will be
    changed for each proposal.
    """
    rng: RNG
    factor: float = 0.1
    """
    Step size multiplier

    The tuning parameter could range from 0 to 1, but in the current
    implementation it'll always be between 0.1 and 0.99.  The tuning parameter
    steps are also pretty large, so the tuning parameter isn't very precise in
    the 0.98-0.99 range.  This mulitplier is used to offset the parameter for
    the cases where steps must be smaller, such as in the nucleotide frequency
    vector.
    """
    weight: float = 1

    _tuning: float = field(init=False, default=0.75)

    def __post_init__(self):
        if len(self.param) <= 1:
            raise ValueError("`param` must have at least two dimensions")

    def propose(self) -> Proposal:
        rng = self.rng

        delta = rng.random_float() * (1 - self._tuning) / 10

        dim_1 = rng.random_int(0, len(self.param))
        dim_2 = dim_1
        while dim_2 == dim_1:
            dim_1 = rng.random_int(0, len(self.param))

        self.param[dim_1] -= delta
        self.param[dim_2] += delta

        # The move is symmetrical, so the Hastings ratio is 0
        return Proposal.Hastings(0)

    def parameters(self) -> list[Parameter]:
        return [self.param]

    def set_tuning(self, parameter: float) -> None:
        self._tuning = parameter

    def get_tuning(self) -> float:
        return self._tuning


@dataclass(slots=True)
class DeltaExchangeInt(Operator):
    """Scales a multidimensional integer parameter without changing its sum

    This operator is analogous to BEAST's `DeltaExchangeOperator` with
    `integer="true"`.  See `DeltaExchange` for details on how dimensions are
    picked.
    """

    param: IntVector
    """
    The multidimensional parameter to edit.  Two random dimensions ones will be
    changed for each proposal.
    """
    rng: RNG
    delta: int = 1
    """
    Step size limit

    The operator will pick a random integer in `[1, delta]`.
    """
    weight: float = 1

    def __post_init__(self):
        if len(self.param) <= 1:
            raise ValueError("`param` must have at least two dimensions")

    def propose(self) -> Proposal:
        rng = self.rng

        delta = rng.random_int(1, self.delta + 1)

        dim_1 = rng.random_int(0, len(self.param))
        dim_2 = dim_1
        while dim_2 == dim_1:
            dim_1 = rng.random_int(0, len(self.param))

        self.param[dim_1] -= delta
        self.param[dim_2] += delta

        # see `DeltaExchange`
        return Proposal.Hastings(0)

    def parameters(self) -> list[Parameter]:
        return [self.param]
