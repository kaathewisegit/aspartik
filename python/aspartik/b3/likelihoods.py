"""
Felsenstein's tree likelihood calculators.
"""

from typing import Protocol

from .._aspartik_rust_impl._b3_rust_impl import (
    CPU4Likelihood as CPU4Likelihood,
    CUDALikelihood as CUDALikelihood,
    HeteroLikelihood as HeteroLikelihood,
    Parallel4Likelihood as Parallel4Likelihood,
)
from . import Stateful


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
