from math import log

from b3 import State, Parameter


class Distribution:
    def __init__(self, param: Parameter, distribution):
        self.param = param

        if hasattr(distribution, "pdf"):
            self.distr_prob = distribution.pdf
        elif hasattr(distribution, "pmf"):
            self.distr_prob = distribution.pmf
        else:
            raise Exception("not a distribution")

    def probability(self, state: State) -> float:
        out = 0

        for i in range(len(self.param)):
            out += log(self.distr_prob(self.param[i]))

        return out
