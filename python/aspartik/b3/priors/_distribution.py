from dataclasses import dataclass

from ...distributions import Continuous
from .. import Prior
from ..parameters import MRCA, Real, RealVector, Root


@dataclass(slots=True)
class Distribution(Prior):
    """
    Calculates prior probability of a single-dimensional parameter according to
    a distribution.
    """

    param: Real | Root | MRCA | RealVector
    distribution: Continuous
    """Distribution against which the parameter prior is calculated."""

    def probability(self) -> float:
        distribution = self.distribution
        param = self.param
        if isinstance(param, RealVector):
            return sum(distribution.ln_pdf(param[i]) for i in range(len(param)))
        else:
            return distribution.ln_pdf(float(param))

    def is_changed(self) -> bool:
        return self.param.is_changed()
