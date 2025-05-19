from .._aspartik_rust_impl import _stats_rust_impl
from typing import Protocol


# fmt: off
class Continuous[T](Protocol): ...
class ContinuousCDF[T](Continuous[T], Protocol): ...
class Discrete[T](Protocol): ...
class DiscreteCDF[T](Discrete[T], Protocol): ...
class Distribution(Protocol): ...
class Sample[T](Protocol): ...
# fmt: on


# ruff: noqa: F822
__all__ = [
    # Classes
    "Gamma",
    "GammaError",
    "Poisson",
    "PoissonError",
    "Uniform",
    "UniformError",
    "Exp",
    "ExpError",
    "LogNormal",
    "LogNormalError",
]

for item in __all__:
    locals()[item] = getattr(_stats_rust_impl.distributions, item)  # type: ignore


def __dir__():
    return __all__
