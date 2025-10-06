from dataclasses import dataclass, field

from ...rng import RNG
from ...stats.distributions import Uniform
from .. import Operator, Proposal, Real, Tree
from ._util import scale_on_range


@dataclass(slots=True)
class RandomWalk(Operator):
    """
    TODO
    """

    param: Real
    window: float
    rng: RNG
    weight: float = 1

    _dist: Uniform = field(default_factory=lambda: Uniform(0, 1), init=False)

    def propose(self) -> Proposal:
        diff, scale = scale_on_range(0, self.window, self._dist, self.rng)
        if self.rng.random_bool():
            diff *= -1

        # TODO: multidimensional parameters
        self.param[0] = self.param[0] + diff

        return Proposal.Hastings(scale)
