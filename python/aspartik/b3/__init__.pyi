from __future__ import annotations
from typing import List, Any, Optional, Sequence, Tuple, Protocol, runtime_checkable
from collections.abc import Iterator

from ..rng import RNG
from .tree import Node, Leaf, Internal
from ..data import DNASeq

class tree: ...

__all__: List[str]

class Tree:
    def __init__(self, names: Sequence[str], rng: RNG): ...
    def update_edge(self, edge: int, new_child: Node) -> None: ...
    def update_weight(self, node: Node, weigth: float) -> None: ...
    def update_root(self, node: Node) -> None: ...
    def swap_parents(self, a: Node, b: Node) -> None: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_internals(self) -> int: ...
    @property
    def num_leaves(self) -> int: ...
    def is_internal(self, node: Node) -> bool: ...
    def is_leaf(self, node: Node) -> bool: ...
    def as_internal(self, node: Node) -> Optional[Internal]: ...
    def as_leaf(self, node: Node) -> Optional[Leaf]: ...
    def root(self) -> Internal: ...
    def weight_of(self, node: Node) -> float: ...
    def children_of(self, node: Internal) -> Tuple[Node, Node]: ...
    def edge_index(self, child: Node) -> int: ...
    def edge_distance(self, edge: int) -> float: ...
    def parent_of(self, node: Node) -> Optional[Internal]: ...
    def is_grandparent(self, node: Internal) -> bool: ...
    def random_node(self, rng: RNG) -> Node: ...
    def random_internal(self, rng: RNG) -> Internal: ...
    def random_leaf(self, rng: RNG) -> Leaf: ...
    def nodes(self) -> Iterator[Node]: ...
    def internals(self) -> Iterator[Internal]: ...
    def verify(self) -> None: ...
    def newick(self) -> str: ...

class Proposal:
    @staticmethod
    def Reject() -> Proposal: ...
    @staticmethod
    def Hastings(ratio: float) -> Proposal: ...
    @staticmethod
    def Accept() -> Proposal: ...

class Parameter:
    """An arbitrary multi-dimensional parameter

    These are used anywhere a generic parameter might be needed.  What makes
    `Parameter` different from regular variables variables is that, when passed
    in the `MCMC` constructor, it's tracked that runtime class.  So, parameter
    values are rolled back when a move is rejected and kept when a move is
    accepted.  This means priors, operators, and likelihood models can use them
    without having to implement state tracking themselves.
    """

    def __len__(self) -> int:
        """Number of dimensions of the parameter"""

    def __getitem__(self, index: int) -> int | float | bool:
        """Value of the index-th dimension

        The type will depend on whenever the parameter was created with `Real`,
        `Integer`, or `Boolean` constructor.
        """

    def __setitem__(self, index: int, value: int | float | bool):
        """Set a new value of the index-th dimension

        It is an error to use a value of a type different from that used in the
        constructor.
        """

    @staticmethod
    def Real(*args: float) -> Any: ...
    @staticmethod
    def Integer(*args: int) -> Any: ...
    @staticmethod
    def Boolean(*args: bool) -> Any: ...
    def is_real(self) -> bool: ...
    def is_integer(self) -> bool: ...
    def is_boolean(self) -> bool: ...

class Likelihood:
    def __init__(
        self,
        sequences: Sequence[DNASeq],
        substitution: Any,
        tree: Tree,
        use_gpu: bool = False,
    ): ...

@runtime_checkable
class Prior(Protocol):
    def probability(self) -> float:
        """Calculates the log prior probability of the model state

        The return value must be a **natural logarithm** of the probability.

        It is presumed that the prior will store all the references to
        parameters and trees it needs for its calculations by itself.
        """

class Operator(Protocol):
    def propose(self) -> Proposal:
        """Proposes a new MCMC step

        It is presumed that the operator will store all the references to
        parameters and trees it wants to edit and will change them accordingly.
        If a move cannot be proposed for any reason `Proposal.Reject` should be
        returned.  MCMC will deal with rolling back the state.
        """

    @property
    def weigth(self) -> float:
        """Influences the probability of the operator being picked

        On each step `MCMC` picks a random operator from the list passed to it.
        It uses this value to weight them.  So, the larger it is, the more
        often the operator will be picked, and visa versa.  This value is read
        once on startup.  So if it's changed mid-execution the old value will
        still be used.
        """

class Logger(Protocol):
    @property
    def every(self) -> int:
        """How often a logger should be called

        The `MCMC` will call each logger when `index % every` is 0.  This value
        is read once when MCMC is created, so if it's changed during execution,
        the old `every` value will continue to be used.
        """

    def log(self, mcmc: MCMC, index: int) -> None:
        """Logging step

        Allows the logger to perform arbitrary actions.
        """

class MCMC:
    def __init__(
        self,
        burnin: int,
        length: int,
        trees: Sequence[Tree],
        params: Sequence[Parameter],
        priors: Sequence[Prior],
        operators: Sequence[Operator],
        likelihoods: Sequence[Likelihood],
        loggers: Sequence[Logger],
        rng: RNG,
    ): ...
    def run(self) -> None: ...
    @property
    def posterior(self) -> float:
        """Posterior probability for the last accepted step"""

    @property
    def likelihood(self) -> float:
        """Total likelihood for the last accepted step"""

    @property
    def prior(self) -> float:
        """Prior likelihood for the current step

        This will trigger a recalculation on all priors.
        """
