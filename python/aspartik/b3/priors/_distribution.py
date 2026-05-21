from dataclasses import dataclass

from ...stats.distributions import Continuous
from .. import Prior
from ..parameters import Real


@dataclass(slots=True)
class Distribution(Prior):
    """
    Calculates prior probability of a single-dimensional parameter according to
    a distribution.
    """

    param: Real
    distribution: Continuous
    """Distribution against which the parameter prior is calculated."""

    def probability(self) -> float:
        return self.distribution.ln_pdf(float(self.param))

    def is_changed(self) -> bool:
        return self.param.is_changed()
