from math import inf
from dataclasses import dataclass

from .. import Parameter


@dataclass
class Bound:
    """A prior which puts limits on the value of a parameter

    This prior serves the same purpose as the `lower` and `upper` attributes on
    BEAST parameters.  It will return `1` if all dimensions of the parameter lie
    within `[lower, upper)` or cancel the proposal by returning negative
    infinity otherwise.

    Due to how the internals of `b3` work, these priors should be first in the
    `priors` list in `run`, to avoid calculating other priors and likelihood if
    the bounds aren't satisfied.
    """

    param: Parameter
    """The parameter to be constrained."""
    lower: int | float = 0
    """Minimum possible value of the parameter, inclusive."""
    upper: int | float = inf
    """Maximum value of the parameter, exclusive (strictly compared)."""

    def probability(self) -> float:
        for i in range(len(self.param)):
            if not (self.lower <= self.param[i] < self.upper):
                return -inf
        return 1
