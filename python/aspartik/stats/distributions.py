from typing import Optional, Protocol, runtime_checkable

from .._aspartik_rust_impl import _stats_rust_impl
from ..rng import RNG


@runtime_checkable
class Continuous[T](Protocol):
    def pdf(self, x: T) -> float: ...
    def ln_pdf(self, x: T) -> float: ...


@runtime_checkable
class ContinuousCDF[T](Continuous[T], Protocol):
    def cdf(self, x: T) -> float: ...
    def sf(self, x: T) -> float: ...
    def inverse_cdf(self, p: float) -> T: ...
    @property
    def lower(self) -> T: ...
    @property
    def upper(self) -> T: ...


@runtime_checkable
class Discrete[T](Protocol):
    def pmf(self, x: T) -> float: ...
    def ln_pmf(self, x: T) -> float: ...


@runtime_checkable
class DiscreteCDF[T](Discrete[T], Protocol):
    def cdf(self, x: T) -> float: ...
    def sf(self, x: T) -> float: ...
    def inverse_cdf(self, p: float) -> T: ...
    @property
    def lower(self) -> T: ...
    @property
    def upper(self) -> T: ...


@runtime_checkable
class Distribution(Protocol):
    def mean(self) -> Optional[float]: ...
    def median(self) -> Optional[float]: ...
    def variance(self) -> Optional[float]: ...
    def std_dev(self) -> Optional[float]: ...
    def entropy(self) -> Optional[float]: ...
    def skewness(self) -> Optional[float]: ...


@runtime_checkable
class Sample[T](Protocol):
    def sample(self, rng: RNG) -> T: ...


# ruff: noqa: F822
__all__ = [
    # Classes
    "Beta",
    "BetaError",
    "Exp",
    "ExpError",
    "Gamma",
    "GammaError",
    "InverseGamma",
    "InverseGammaError",
    "Laplace",
    "LaplaceError",
    "LogNormal",
    "LogNormalError",
    "Normal",
    "NormalError",
    "Poisson",
    "PoissonError",
    "Uniform",
    "UniformError",
    # protocols
    "Continuous",
    "ContinuousCDF",
    "Discrete",
    "DiscreteCDF",
    "Distribution",
    "Sample",
]

for item in __all__[:18]:
    locals()[item] = getattr(_stats_rust_impl.distributions, item)


def __dir__():
    return __all__
