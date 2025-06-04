from dataclasses import dataclass

from .. import Real, Integer, Prior
from ...stats import distributions


@dataclass
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
    distribution: distributions.Distribution
    """Distribution against which the parameter prior is calculated."""

    def __post_init__(self):
        # TODO: check that param type fits the distr type
        if hasattr(self.distribution, "pdf"):
            self.distr_prob = self.distribution.ln_pdf  # type: ignore
        elif hasattr(self.distribution, "pmf"):
            self.distr_prob = self.distribution.ln_pmf  # type: ignore
        else:
            raise Exception("not a distribution")

    def probability(self) -> float:
        """
        For multi-dimensional parameters the sum of log probabilities of all
        dimensions is returned.
        """

        out = 0

        for i in range(len(self.param)):
            out += self.distr_prob(self.param[i])

        return out
