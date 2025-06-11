from __future__ import annotations
from typing import (
    List,
    Any,
    Optional,
    Sequence,
    Tuple,
    Protocol,
    runtime_checkable,
    Literal,
)
from collections.abc import Iterator

from ..rng import RNG
from .tree import Node, Leaf, Internal
from ..data import DNASeq

class tree: ...

__all__: List[str]

class Stateful(Protocol):
    def accept(self) -> None: ...
    def reject(self) -> None: ...

class Tree(Stateful):
    """A phylogenetic tree

    Unlike in BEAST2, `Tree` is a self-contained typed which all of the
    topological data.  What this means is that node types (`Internal` and
    `Leaf`) are simply indexes into the `Tree`.  So, all operations, like
    getting parents of node weights have to go through `Tree`'s methods.

    The current implementation of `Tree` only supports bifurcating topologies.
    """

    def __init__(self, names: Sequence[str], rng: RNG):
        """
        - `names` is the list of names of leaf nodes.
        - `rng` is used to build a random tree.
        """

    def update_edge(self, edge: int, new_child: Node) -> None:
        """Sets the **child** of `edge` to `new_child`

        This will only change the child, so the parent (internal node from
        which `edge` comes out) will now have `node` as a child.

        This function doesn't do any validation, it's up to the operator to
        preserve the validity of the tree.
        """
    def update_weight(self, node: Node, weigth: float) -> None:
        """Sets the weight of `node` to `weight`"""
    def update_root(self, node: Node) -> None:
        """Makes `node` the root of the tree

        As the topology can be temporarily broken while the edges are being
        swapped, `Tree` can't automatically figure out which node is the root
        one.  So, operators which change the root of the tree have to update it
        manually.
        """
    def swap_parents(self, a: Node, b: Node) -> None:
        """Swaps the parents of nodes `a` and `b`

        `a` and `b` must not be a child/parent and neither of them can be a
        root node.  If `a` and `b` share the same parent, they switch polarity
        (left child becomes the right child and visa versa).
        """
    @property
    def num_nodes(self) -> int:
        """The total number of nodes in the tree"""
    @property
    def num_internals(self) -> int:
        """The number of internal nodes (those with children)"""
    @property
    def num_leaves(self) -> int:
        """The number of leaf nodes"""
    def is_internal(self, node: Node) -> bool:
        """Returns `True` if the node is internal"""
    def is_leaf(self, node: Node) -> bool: ...
    def as_internal(self, node: Node) -> Optional[Internal]:
        """
        Converts `node` to the type `Internal` if it is internal, or returns
        `None` otherwise
        """
    def as_leaf(self, node: Node) -> Optional[Leaf]: ...
    def root(self) -> Internal:
        """Returns the root node of the tree

        Note that the root node might change after tree has been edited, so the
        returned node is only guaranteed to be root as long as the tree hasn't
        been edited.
        """
    def weight_of(self, node: Node) -> float:
        """Returns the weight of `node`

        Weight here means node's age in some unlabeled units.
        """
    def children_of(self, node: Internal) -> Tuple[Node, Node]:
        """Returns a tuple of the left and right children of `node`

        This function takes the `Internal` type as its input, so it is
        guaranteed to always return the children.  See `as_internal` for
        converting general nodes to internal ones.
        """
    def edge_index(self, child: Node) -> int:
        """Returns the index of an edge from `child` to its parent"""
    def edge_distance(self, edge: int) -> float:
        """Returns the length of `edge`

        The length is the distance between the parent and the child nodes of
        that edge.
        """
    def parent_of(self, node: Node) -> Optional[Internal]:
        """Returns the parent of `node`, or `None` for the root node"""
    def is_grandparent(self, node: Internal) -> bool:
        """Returns `True` if both children of this node are also internal"""
    def num_grandparents(self) -> int:
        """Number of nodes for whom `is_grandparent` returns `True`"""
    def random_node(self, rng: RNG) -> Node:
        """Returns a random node from the tree

        It can be both an internal node or a leaf.  See `random_internal` and
        `random_leaf` for getting a random node of a specific kind.
        """
    def random_internal(self, rng: RNG) -> Internal:
        """Returns a random internal node"""
    def random_leaf(self, rng: RNG) -> Leaf:
        """Returns a random leaf node"""
    def nodes(self) -> Iterator[Node]:
        """An iterator over all of trees nodes

        All of the `Leaf` nodes go before `Internal` ones.
        """
    def internals(self) -> Iterator[Internal]:
        """An iterator over all of the trees internal nodes"""
    def leaves(self) -> Iterator[Leaf]:
        """An iterator over all of the trees leaf nodes"""
    def verify(self) -> None:
        """Throws an exception if a tree is malformed

        This function ensures that:

        - No leaf has become anyone's parent.
        - All parent nodes are older than their children.
        - Parents match their children (mismatches can happen when
          `update_edge` is used incorrectly).
        - There's only one root (two or more can be set with `update_root`).
        - The tree is a tree, meaning that topologically it has no cycles and
          is connected.
        """
    def newick(self) -> str:
        """Returns the tree topology in the Newick format

        Leaf nodes will be labeled with the names passed to the constructor
        while the internal nodes are unlabeled.
        """

class Proposal:
    """A result of the move proposed by an operator

    While the operators edit the tree directly, they need to communicate the
    status of their move to `MCMC`.  This is the class used for that.
    """

    @classmethod
    def Reject(cls) -> Proposal:
        """Aborts the move unconditionally

        All of the trees and parameters are rolled back.  This is relatively
        fast, as it typically skips recalculating the likelihoods.
        """
    @classmethod
    def Hastings(cls, ratio: float) -> Proposal:
        """Proposes the move with the `ratio`

        This is the ratio from the Metropolis–Hastings algorithm.
        """
    @classmethod
    def Accept(cls) -> Proposal:
        """Accepts the move unconditionally"""

class Real(Stateful):
    def __init__(self, *values: float): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def __float__(self) -> float: ...
    # richcmp
    def __lt__(self, other: float | Real) -> bool: ...
    def __le__(self, other: float | Real) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: float | Real) -> bool: ...
    def __ge__(self, other: float | Real) -> bool: ...

class Integer(Stateful):
    def __init__(self, *values: int): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...
    def __setitem__(self, index: int, value: int) -> None: ...
    def __int__(self) -> int: ...
    # richcmp
    def __lt__(self, other: int | Integer) -> bool: ...
    def __le__(self, other: int | Integer) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: int | Integer) -> bool: ...
    def __ge__(self, other: int | Integer) -> bool: ...

class Boolean(Stateful):
    def __init__(self, *values: bool): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> bool: ...
    def __setitem__(self, index: int, value: bool) -> None: ...
    # richcmp
    def __lt__(self, other: bool | Boolean) -> bool: ...
    def __le__(self, other: bool | Boolean) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other: bool | Boolean) -> bool: ...
    def __ge__(self, other: bool | Boolean) -> bool: ...

type Parameter = Real | Integer | Boolean

class Likelihood:
    def __init__(
        self,
        sequences: Sequence[DNASeq],
        # TODO: types
        substitution: Any,
        clock: Any,
        tree: Tree,
        calculator: Literal["cpu", "thread", "cuda"] = "cpu",
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
    every: int
    """How often a logger should be called

    The `MCMC` will call each logger when `index % every` is 0.  This value
    is read once when MCMC is created, so if it's changed during execution,
    the old `every` value will continue to be used.
    """

    def log(self, mcmc: MCMC) -> None:
        """Logging step

        Allows the logger to perform arbitrary actions.
        """

class MCMC:
    def __init__(
        self,
        burnin: int,
        length: int,
        state: Sequence[Stateful],
        priors: Sequence[Prior],
        operators: Sequence[Operator],
        likelihoods: Sequence[Likelihood],
        loggers: Sequence[Logger],
        rng: RNG,
        *,
        validate: bool = False,
        cuda_device: int = 0,
        thread_split_size: int = 400,
    ): ...
    @property
    def current_step(self) -> int: ...
    @property
    def state(self) -> List[Stateful]: ...
    @property
    def priors(self) -> List[Prior]: ...
    @property
    def likelihoods(self) -> List[Likelihood]: ...
    @property
    def loggers(self) -> List[Logger]: ...
    @property
    def rng(self) -> RNG: ...
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

    def run(self) -> None: ...
