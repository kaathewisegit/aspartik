from math import inf
from dataclasses import dataclass

from .. import Real, Prior


@dataclass
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

    param: Real
    """The parameter to be constrained."""
    lower: float = 0
    """Minimum possible value of the parameter, inclusive."""
    upper: float = inf
    """Maximum value of the parameter, exclusive (strictly compared)."""

    def probability(self) -> float:
        if not (self.lower <= self.param < self.upper):
            return -inf
        else:
            return 1
