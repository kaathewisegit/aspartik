from dataclasses import dataclass
from math import log

from ...stats.distributions import Gamma
from .. import Prior
from ..parameters import RealVector


@dataclass(slots=True)
class MarkovChainDistribution(Prior):
    param: RealVector
    shape: float = 1.0

    def probability(self) -> float:
        out = -log(self.param[0])

        for i in range(1, len(self.param)):
            mean = self.param[i - 1]
            x = self.param[i]

            scale = mean / self.shape
            gamma = Gamma(self.shape, 1 / scale)

            out += gamma.ln_pdf(x)

        return out

    def is_changed(self) -> bool:
        return self.param.is_changed()
