from dataclasses import dataclass

from .._aspartik_rust_impl._b3_rust_impl import (
    CPU4Likelihood as CPU4Likelihood,
    CUDALikelihood as CUDALikelihood,
    Thread4Likelihood as Thread4Likelihood,
)
from . import Stateful
from .parameters import Weights


class Likelihood(Stateful):
    """
    Tree likelihood calculator

    This object calculates the likelihood of a tree given the sequence data
    using Felsenstein's tree pruning algorithm.

    There are several implementations, each with its own options (**TODO**
    docs).
    """

    def propose(self) -> None: ...
    def likelihood(self) -> float: ...
    def cached_likelihood(self) -> float: ...


@dataclass(slots=True)
class WeightedLikelihood(Likelihood):
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

    def cached_likelihood(self) -> float:
        out = 0
        for likelihood, weight in zip(self.likelihoods, self.weights):
            out += likelihood.cached_likelihood() * weight
        return out

    def accept(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.accept()

    def reject(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.reject()
