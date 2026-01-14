from typing import Optional, Protocol, runtime_checkable

from .._aspartik_rust_impl._stats_rust_impl import (
    Beta as Beta,
    BetaError as BetaError,
    Exp as Exp,
    ExpError as ExpError,
    Gamma as Gamma,
    GammaError as GammaError,
    InverseGamma as InverseGamma,
    InverseGammaError as InverseGammaError,
    Laplace as Laplace,
    LaplaceError as LaplaceError,
    LogNormal as LogNormal,
    LogNormalError as LogNormalError,
    Normal as Normal,
    NormalError as NormalError,
    Poisson as Poisson,
    PoissonError as PoissonError,
    Uniform as Uniform,
    UniformError as UniformError,
)
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
