from math import log

from .. import State, Parameter


class Distribution:
    """Calculates prior probability of a parameter according to a distribution."""

    def __init__(self, param: Parameter, distribution):
        """
        Args:
            param:
                Parameter to estimate.  Can be either `Real` or `Integer` for
                discrete distributions.
            distribution:
                Distribution against which the parameter prior is calculated.
        """

        self._param = param

        if hasattr(distribution, "pdf"):
            self.distr_prob = distribution.pdf
        elif hasattr(distribution, "pmf"):
            self.distr_prob = distribution.pmf
        else:
            raise Exception("not a distribution")

    def probability(self, state: State) -> float:
        """
        For multi-dimensional parameters the sum of log probabilities of all
        dimensions is returned.
        """

        out = 0

        for i in range(len(self._param)):
            out += log(self.distr_prob(self._param[i]))

        return out
