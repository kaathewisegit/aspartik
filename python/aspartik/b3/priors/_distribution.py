from dataclasses import dataclass, field
from typing import SupportsFloat

from ...stats.distributions import Continuous, Discrete
from .. import Prior
from ..parameters import Integer, Real


@dataclass(slots=True)
class Distribution(Prior):
    """
    Calculates prior probability of a single-dimensional parameter according to
    a distribution.
    """

    param: SupportsFloat
    distribution: Continuous
    """Distribution against which the parameter prior is calculated."""

    def probability(self) -> float:
        return self.distribution.ln_pdf(float(self.param))
