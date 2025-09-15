from typing import Sequence, SupportsFloat, SupportsInt

from aspartik.stats.distributions import Distribution as StatsDistribution

from .. import Leaf, Prior, Tree

class ConstantPopulation(Prior):
    def __init__(self, tree: Tree, population: SupportsFloat): ...

class Distribution(Prior):
    def __init__(
        self,
        param: SupportsInt | SupportsFloat,
        distribution: StatsDistribution,
    ): ...

class Monophyly(Prior):
    """
    Ensures that a group of leaves form a monophyly

    Returns a static probability if the specified leaves are monophyletic or
    aborts the move otherwise.
    """

    def __init__(self, tree: Tree, leaves: Sequence[Leaf]): ...

class Yule(Prior):
    def __init__(self, tree: Tree, birth_rate: SupportsFloat): ...
