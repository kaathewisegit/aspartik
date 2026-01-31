from dataclasses import dataclass

from ...rng import RNG
from .. import Operator, Proposal
from ..parameters import RealVector


@dataclass(slots=True)
class DeltaExchange(Operator):
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
    factor: float
    """
    The move size is a random value between 0 and 1 multiplied by `factor`.
    """
    rng: RNG
    weight: float = 1

    def __post_init__(self):
        if len(self.param) <= 1:
            raise ValueError(f"`param` must have at least two dimensions")

    def propose(self) -> Proposal:
        rng = self.rng

        delta = rng.random_float() * self.factor

        dim_1 = rng.random_int(0, len(self.param))
        dim_2 = dim_1
        while dim_2 == dim_1:
            dim_1 = rng.random_int(0, len(self.param))

        self.param[dim_1] -= delta
        self.param[dim_2] += delta

        # The move is symmetrical, so the Hastings ratio is 0
        return Proposal.Hastings(0)
