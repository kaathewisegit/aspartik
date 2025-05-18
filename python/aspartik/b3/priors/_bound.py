from math import inf


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

    def __init__[T: int | float](self, param, lower: T = 0, upper: T = inf):
        """Creates a bound prior

        Args:
            param: The parameter to be constrained.
            lower: Minimum value of the parameter.
            upper: Maximum value of the parameter, strictly compared.
        """
        self.param = param
        self.lower = lower
        self.upper = upper

    def probability(self) -> float:
        if self.lower <= self.param < self.upper:
            return 1
        else:
            return -inf
