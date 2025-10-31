from collections.abc import Sequence
from dataclasses import dataclass
from typing import SupportsFloat

from ..b3 import Operator, Prior, Stateful, Tree
from ..b3.parameters import Scalable
from ..b3.tree import Leaf
from ..rng import RNG
from ..stats.distributions import Distribution, Sample

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

@dataclass(slots=True)
class SubtreeLeap(Operator):
    """
    Moves a node a distance, changing the topology randomly

    First, a distance delta is sampled from the distribution.  The operator
    selects a random node and all edges `delta` away from that node (down if
    the delta is negative or up and down if it's positive).  One of those edges
    is randomly selected and the node is spliced into it.  If the delta is
    above the root, the node will become the new root.
    """

    tree: Tree
    """The tree to edit."""
    distribution: Sample[float]
    """The distribution to draw the height move distance from"""
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

class Real(Stateful, Scalable):
    """
    Real multidimensional parameter

    It acts as a list of `float`s, except it implements
    [`Stateful`](#Stateful).  Note that this uses regular 64-bit floating point
    numbers underneath, not some custom arbitrary precision real type.
    """
    def __init__(self, *values: float): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def __float__(self) -> float: ...
    # richcmp
    def __lt__(self, other: float | Real) -> bool: ...
    def __le__(self, other: float | Real) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: float | Real) -> bool: ...
    def __ge__(self, other: float | Real) -> bool: ...

class Integer(Stateful):
    """
    Integer multidimensional parameter

    It acts as a list of `int`s, except it implements [`Stateful`](#Stateful).
    """
    def __init__(self, *values: int): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...
    def __setitem__(self, index: int, value: int) -> None: ...
    def __int__(self) -> int: ...
    # richcmp
    def __lt__(self, other: int | Integer) -> bool: ...
    def __le__(self, other: int | Integer) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: int | Integer) -> bool: ...
    def __ge__(self, other: int | Integer) -> bool: ...

class Boolean(Stateful):
    """
    Boolean multidimensional parameter

    It acts as a list of `bool`s, except it implements [`Stateful`](#Stateful).
    """

    def __init__(self, *values: bool): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> bool: ...
    def __setitem__(self, index: int, value: bool) -> None: ...
    # richcmp
    def __lt__(self, other: bool | Boolean) -> bool: ...
    def __le__(self, other: bool | Boolean) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: bool | Boolean) -> bool: ...
    def __ge__(self, other: bool | Boolean) -> bool: ...

Parameter = Real | Integer | Boolean

@dataclass
class JC:
    """
    Jukes-Cantor

    A simple model with equal state transition rates.

    Jukes and Cantor 1969, Evolution of Protein Molecules,
    <https://doi.org/10.1016/b978-1-4832-3211-9.50009-7>.
    """

@dataclass
class K80:
    """Kimura 80

    Equal base frequencies (A/C/G/T) with different transition (keeps
    purines/pyrimidines) and transversion (purine to pyrimidine and visa
    versa).

    Kimura 1980, A simple method for estimating evolutionary rates of base
    substitutions through comparative studies of nucleotide sequences,
    <https://doi.org/10.1007/BF01731581>.
    """

    kappa: SupportsFloat
    """
    A transition is taken to be kappa times more likely than a transversion.
    """

@dataclass
class HKY:
    """
    Hasegawa et al. 1985

    A model which can be thought of as a combination of K80 and F81: both base
    rates and transition/transversion ratio are configurable.

    Hasegawa et al. 1985, Dating of the human-ape splitting by a molecular
    clock of mitochondrial DNA, <https://doi.org/10.1007/BF02101694>.
    """

    frequencies: Sequence[float]
    kappa: SupportsFloat
