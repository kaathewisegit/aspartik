from .._aspartik_rust_impl._b3_rust_impl import (
    CPU4Likelihood as CPU4Likelihood,
    CUDALikelihood as CUDALikelihood,
    Thread4Likelihood as Thread4Likelihood,
)
from . import Stateful


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
