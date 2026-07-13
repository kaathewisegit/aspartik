from dataclasses import dataclass, field
from math import log

from ...distributions import Sample
from ...rng import RNG
from .. import Operator, Proposal, TunableOperator
from ..parameters import Parameter, Tree
from ._util import sample_range


@dataclass(slots=True)
class RootSlide(Operator, TunableOperator):
    """
    Scales the height of the root node

    The height will be scaled between `factor` and `1 / factor` sampled with
    `distribution`.  If the move will put the root below one of its children
    the operation gets rejected.
    """

    tree: Tree
    """The tree to edit."""
    distribution: Sample[float]
    """The distribution to draw the height move distance from"""
    rng: RNG
    weight: float = 1

    _factor: float = field(init=False, default=0.75)

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng
        root = tree.root

        low, high = self._factor, 1 / self._factor
        scale = sample_range(low, high, self.distribution, rng)

        old_height = tree.height_of(root)

        left, right = tree.children_of(root)
        lower_height = max(tree.height_of(left), tree.height_of(right))

        new_height = (old_height - lower_height) * scale + lower_height

        tree.set_height(root, new_height)

        return Proposal.Hastings(-log(scale))

    def parameters(self) -> list[Parameter]:
        return [self.tree]

    def set_tuning(self, parameter: float) -> None:
        self._factor = parameter

    def get_tuning(self) -> float:
        return self._factor
