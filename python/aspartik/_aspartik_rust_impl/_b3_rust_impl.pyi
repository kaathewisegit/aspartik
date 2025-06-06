from dataclasses import dataclass

from ..b3 import Prior, Tree, Parameter, Operator
from ..rng import RNG
from ..stats.distributions import Distribution

class Yule(Prior):
    """Uncalibrated Yule birth-rate model"""

    def __init__(self, tree: Tree, birth_rate: Parameter): ...

class ConstantPopulation(Prior):
    """Constant population coalescent"""

    def __init__(self, tree: Tree, population: Parameter): ...

@dataclass
class TreeScale(Operator):
    """Scales the age of the entire tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale all internal nodes by a random scale which is randomly
    picked depending on `factor` and `distribution`.
    """

    tree: Tree
    """The tree to scale."""
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
class EpochScale(Operator):
    """Scales a random epoch in a tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the weight of 0).
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
