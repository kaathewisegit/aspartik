from typing import Protocol, runtime_checkable


from .._aspartik_rust_impl import _b3_rust_impl


for item in ["Likelihood", "Parameter", "Proposal", "MCMC", "Tree"]:
    locals()[item] = getattr(_b3_rust_impl, item)


@runtime_checkable
class Prior(Protocol):
    def probability(self): ...


@runtime_checkable
class Operator(Protocol):
    def propose(self): ...
    @property
    def weigth(self): ...


@runtime_checkable
class Logger(Protocol):
    every: int

    def log(self, mcmc, index): ...


__all__ = [
    # Rust
    "Likelihood",
    "Parameter",
    "Proposal",
    "MCMC",
    "Tree",
    # Rust submodules
    "tree",
    # Protocols
    "Prior",
    "Operator",
    "Logger",
    # Python
    "loggers",
    "operators",
    "priors",
    "substitutions",
]


def __dir__():
    return __all__
