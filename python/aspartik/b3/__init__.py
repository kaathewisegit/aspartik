# ruff: noqa: E402

from typing import Protocol, runtime_checkable


@runtime_checkable
class Prior(Protocol):
    def probability(self) -> float: ...


from .._aspartik_rust_impl import _b3_rust_impl

for item in ["Likelihood", "Parameter", "Proposal", "MCMC", "Tree"]:
    locals()[item] = getattr(_b3_rust_impl, item)

from . import loggers, operators, priors, substitutions
from . import tree


__all__ = [
    # Rust
    "Likelihood",
    "Parameter",
    "Proposal",
    "MCMC",
    "Tree",
    # Rust submodules
    "tree",
    # Python
    "loggers",
    "operators",
    "priors",
    "substitutions",
]


def __dir__():
    return __all__
