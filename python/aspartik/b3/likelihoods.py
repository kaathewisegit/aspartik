"""
Felsenstein's tree likelihood calculators.
"""

from dataclasses import dataclass
from typing import Protocol

from .._aspartik_rust_impl._b3_rust_impl import (
    CPU4Likelihood as CPU4Likelihood,
    CUDALikelihood as CUDALikelihood,
    Thread4Likelihood as Thread4Likelihood,
)
from . import Stateful
from .parameters import Weights


class Likelihood(Stateful, Protocol):
    """
    Tree likelihood calculator

    This object calculates the likelihood of a tree given the sequence data
    using Felsenstein's tree pruning algorithm.

    There are several implementations, each with its own options.
    """

    def propose(self) -> None:
        """
        Fetches the current model state and starts the calculations.

        Each implementations are responsible for pulling the state.

        Depending on the implementation, `propose` might start the calculation
        in parallel and return.  In this case `likelihood` will block on it.
        """
        ...

    def likelihood(self) -> float:
        """
        Returns the tree likelihood calculated.

        If `propose` was called during this epoch, this must the likelihood as
        a result of the proposal.  If `accept` or `reject` was called last,
        this method must return the last accepted likelihood.
        """
        ...


@dataclass(slots=True)
class CompoundLikelihood(Likelihood):
    """
    Combines several likelihoods.

    `CompoundLikelihood` doesn't use a thread pool, but depending on the
    `Likelihood` implementers, calculations might be done in parallel.
    """

    likelihoods: list[Likelihood]

    def propose(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.propose()

    def likelihood(self) -> float:
        return sum(l.likelihood() for l in self.likelihoods)

    def accept(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.accept()

    def reject(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.reject()


@dataclass(slots=True)
class WeightedLikelihood(Likelihood):
    """
    Combines several likelihoods with dynamic weights.

    The formula is `Σ l_i * w_i`.  It is inspired [GHOST] and can be used to
    analyse several sequence evolution models.  Much like `CompoundLikelihood`,
    most likelihood implementations will be

    [GHOST]: https://doi.org/10.1093/sysbio/syz051
    """

    likelihoods: list[Likelihood]
    weights: Weights

    def propose(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.propose()

    def likelihood(self) -> float:
        out = 0
        for likelihood, weight in zip(self.likelihoods, self.weights):
            out += likelihood.likelihood() * weight
        return out

    def accept(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.accept()

    def reject(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.reject()
