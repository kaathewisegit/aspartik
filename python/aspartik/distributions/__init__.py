from typing import Optional, Protocol, SupportsFloat, runtime_checkable

from .._aspartik_rust_impl._distributions_rust_impl import (
    Beta as Beta,
    Exponential as Exponential,
    Gamma as Gamma,
    InverseGamma as InverseGamma,
    Laplace as Laplace,
    LogNormal as LogNormal,
    Normal as Normal,
    Uniform as Uniform,
)
from ..rng import RNG


@runtime_checkable
class Continuous(Protocol):
    """
    A continuous statistical distribution

    In practice this defines the `pdf` and `ln_pdf` methods.
    """

    def pdf(self, x: SupportsFloat) -> float:
        """
        Probability density function

        May throw an exception if `x` is outside of support.
        """
        ...

    def ln_pdf(self, x: SupportsFloat) -> float:
        """
        Logarithm of a probability density function

        Some implementations might be more precise than calculating a logarithm
        on the result of the `pdf` method.
        """
        ...

    def cdf(self, x: SupportsFloat) -> float:
        """
        Continuous distribution function

        This method might throw an exception if `x` is outside of the supported
        interval.
        """
        ...

    def sf(self, x: SupportsFloat) -> float:
        """Survival function"""
        ...

    def inverse_cdf(self, p: SupportsFloat) -> float:
        """
        Inverse continuous distribution function

        `p` must be in the interval `[0, 1]`.
        """
        ...

    @property
    def lower(self) -> float:
        """
        Lower bound of support

        `-inf` for distributions defined on all real numbers.
        """
        ...

    @property
    def upper(self) -> float:
        """
        Upper bound of support

        Will be `+inf` for distributions defined on the entire line or on all
        positive reals.
        """
        ...


@runtime_checkable
class Discrete[T](Protocol):
    def pmf(self, x: T) -> float: ...
    def ln_pmf(self, x: T) -> float: ...
    def cdf(self, x: T) -> float: ...
    def sf(self, x: T) -> float: ...
    def inverse_cdf(self, p: SupportsFloat) -> T: ...
    @property
    def lower(self) -> T: ...
    @property
    def upper(self) -> T: ...


@runtime_checkable
class Statistics(Protocol):
    def mean(self) -> Optional[float]: ...
    def median(self) -> Optional[float]: ...
    def variance(self) -> Optional[float]: ...
    def std_dev(self) -> Optional[float]: ...
    def entropy(self) -> Optional[float]: ...


@runtime_checkable
class Sample[T](Protocol):
    def sample(self, rng: RNG) -> T: ...
