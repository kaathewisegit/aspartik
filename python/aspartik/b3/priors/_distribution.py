from dataclasses import dataclass

from .. import Parameter
from ... import stats


@dataclass
class Distribution:
    """Calculates prior probability of a parameter according to a distribution."""

    param: Parameter
    """
    Parameter to estimate.  Can be either `Real` or `Integer` for discrete
    distributions.
    """
    distribution: stats.distributions.Distribution
    """Distribution against which the parameter prior is calculated."""

    def __post_init__(self):
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
