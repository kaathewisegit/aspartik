from typing import Protocol, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    Likelihood as Likelihood,
    Proposal as Proposal,
    MCMC as MCMC,
    Tree as Tree,
    Real as Real,
    Integer as Integer,
    Boolean as Boolean,
    tree as tree,
)


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
