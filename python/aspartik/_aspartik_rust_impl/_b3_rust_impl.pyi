from collections.abc import Sequence
from dataclasses import dataclass
from typing import SupportsFloat

from ..b3 import Operator, Prior, Tree
from ..b3.parameters import Parameter
from ..b3.tree import Leaf
from ..rng import RNG
from ..stats.distributions import Distribution

@dataclass
class EpochScale(Operator):
    """Scales a random epoch in a tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the height of 0).
    """

    tree: Tree
    factor: float
    """
    The scaling ratio will be sampled from `(factor, 1 / factor)`.  So, the
    factor must be between 0 and 1 and the smaller it is the larger the steps
    will be.
    """
    distribution: Distribution
    """Distribution from which the scale is sampled."""
    rng: RNG
    weight: float = 1

@dataclass
class ConstantPopulation(Prior):
    """Constant population coalescent"""

    tree: Tree
    population: SupportsFloat

@dataclass
class ExponentialGrowth(Prior):
    tree: Tree
    population: SupportsFloat
    growth_rate: SupportsFloat

@dataclass
class Monophyly(Prior):
    """
    Ensures that a group of leaves form a monophyly

    Returns a static probability if the specified leaves are monophyletic or
    aborts the move otherwise.
    """

    tree: Tree
    leaves: Sequence[Leaf]

class Yule(Prior):
    """Uncalibrated Yule birth-rate model"""

    tree: Tree
    birth_rate: SupportsFloat
