from dataclasses import dataclass
from math import inf
from typing import Protocol

from .. import Prior
from ..parameters import Real


class IntervalComparable(Protocol):
    def __lt__(self, other: float) -> bool: ...
    def __ge__(self, other: float) -> bool: ...


@dataclass(slots=True)
class Bound(Prior):
    """Puts limits on the value of a parameter

    This prior serves the same purpose as the `lower` and `upper` attributes on
    BEAST parameters.  It will return `1` if all dimensions of the parameter
    lie within `[lower, upper)` or cancel the proposal by returning negative
    infinity otherwise.

    Due to how the internals of `b3` work, these priors should be first in the
    `priors` list in `run`, to avoid calculating other priors and likelihood if
    the bounds aren't satisfied.
    """

    param: IntervalComparable
    """The parameter to be constrained."""
    lower: float = 0
    """Minimum possible value of the parameter, inclusive."""
    upper: float = inf
    """Maximum value of the parameter, exclusive (strictly compared)."""

    def probability(self) -> float:
        if not (self.lower <= self.param < self.upper):
            return -inf
        else:
            return 0
