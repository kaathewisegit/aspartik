from collections.abc import Callable
from dataclasses import dataclass, field

from ...stats.distributions import Continuous, Discrete
from .. import Integer, Prior, Real


@dataclass(slots=True)
class Distribution(Prior):
    """Calculates prior probability of a parameter according to a distribution

    For multidimensional parameters the independent probability of all
    dimensions is calculated.
    """

    param: Real | Integer
    """
    Parameter to estimate.  Can be either `Real` or `Integer` for discrete
    distributions.
    """
    distribution: Continuous | Discrete
    """Distribution against which the parameter prior is calculated."""
    _distr_prob: Callable = field(init=False)

    def __post_init__(self):
        if isinstance(self.distribution, Continuous):
            self._distr_prob = self.distribution.ln_pdf
            if not isinstance(self.param, Real):
                raise TypeError(
                    "Expected the parameter to be `Real` because the distribution is continuous"
                )
        elif isinstance(self.distribution, Discrete):
            if not isinstance(self.param, Integer):
                raise TypeError(
                    "Expected the parameter to be `Integer` because the distribution is discrete"
                )
            self._distr_prob = self.distribution.ln_pmf
        else:
            raise Exception("not a distribution")

    def probability(self) -> float:
        """
        For multi-dimensional parameters the sum of log probabilities of all
        dimensions is returned.
        """

        out = 0

        for i in range(len(self.param)):
            out += self._distr_prob(self.param[i])

        return out
