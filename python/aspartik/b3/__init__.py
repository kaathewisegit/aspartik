from typing import Protocol, runtime_checkable


from .._aspartik_rust_impl import _b3_rust_impl

__all__ = [
    # Rust
    "Likelihood",
    "Proposal",
    "MCMC",
    "Tree",
    "Real",
    "Integer",
    "Boolean",
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


for item in __all__[:8]:
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


@runtime_checkable
class Stateful(Protocol):
    def accept(self): ...
    def reject(self): ...


def __dir__():
    return __all__
