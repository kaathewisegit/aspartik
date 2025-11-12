from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, Optional, Protocol, SupportsFloat

from ..b3 import Callback, Leaf, Node, Operator, Prior, Stateful, Tree
from ..b3.likelihoods import Likelihood
from ..b3.parameters import Scalable
from ..b3.substitutions import HKY, JC, K80
from ..data.msa import MSA
from ..data.newick import Tree as NewickTree
from ..rng import RNG
from ..stats.distributions import Distribution, Sample

class Tree(Stateful):
    """A phylogenetic tree

    Unlike BEAST2, where the tree is implemented as a collection of nodes
    pointing to each other, in `b3` `Tree` is a self-contained data structure
    which holds all of the topology and heights.  This means that nodes
    (`Internal` and `Leaf`) are identifiers of nodes in a given `Tree` object,
    much like indices of an array.  This means that all operations, such as
    getting parents of node heights, have to go through `Tree`'s methods.

    The current implementation of `Tree` only supports bifurcating topologies.
    """

    def __init__(self, names: Sequence[str], rng: RNG):
        """
        - `names` is the list of names of leaf nodes.
        - `rng` is used to build a random tree.

        All heights are initially set to 0.  Use `set_height` for dated tips
        and `set_random_heights` to randomize the positions of internal nodes.
        """

    @classmethod
    def from_newick(_cls, newick: NewickTree) -> Tree:
        """
        Initializes the tree from a Newick object.

        The Newick tree must be strictly bifurcating and all of its edges must
        have a defined length.
        """

    def set_random_edges(self, rng: RNG):
        """
        Randomly sets edges between the nodes

        This methods creates a random [Prüfer sequence][wiki] and rearranges
        the graph according to it.  Note that it will always the internal node
        with the largest index (`num_nodes - 1`) the root.

        [wiki]: https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence
        """

    def set_random_heights(self, diff, rng: RNG):
        """
        Randomizes the heights of internal nodes

        Each internal node gets a height distributed uniformly between `diff`
        and `2 * diff` plus the height of the highest of its children.
        """

    @property
    def names(self) -> list[str]:
        """
        A list of all leaf names.

        The order is the same as leaf indices: the first name is that of
        `Leaf(0)`, the second one is `Leaf(1)`, and so on.
        """

    def scale(self, scale: float):
        """
        Multiplies the heights of all internal nodes by `scale`


        ### Exceptions

        Throws a `RuntimeError` if any of the internal nodes would be moved
        below either of its children.
        """

    def update_edge(self, edge: int, new_child: Node) -> None:
        """Sets the **child** of `edge` to `new_child`

        This will only change the child, so the parent (internal node from
        which `edge` comes out) will now have `node` as a child.

        This function doesn't do any validation, it's up to the operator to
        preserve the validity of the tree.
        """
    def set_height(self, node: Node, weigth: float) -> None:
        """Sets the height of `node` to `height`"""
    def set_root(self, node: Node) -> None:
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
    def is_leaf(self, node: Node) -> bool:
        """Returns `True` if the node is a leaf"""
    def as_internal(self, node: Node) -> Optional[Internal]:
        """
        Converts `node` to the type `Internal` if it is internal, or returns
        `None` otherwise
        """
    def as_leaf(self, node: Node) -> Optional[Leaf]:
        """
        Converts `node` to the type `Leaf` if it is a leaf, or returns
        `None` otherwise
        """
    @property
    def root(self) -> Internal:
        """Returns the root node of the tree

        Note that the root node might change after tree has been edited, so the
        returned node is only guaranteed to be root as long as the tree hasn't
        been edited.
        """
    def height_of(self, node: Node) -> float:
        """Returns the height of `node`

        Height here means node's age in some unlabeled units.
        """
    def children_of(self, node: Internal) -> tuple[Node, Node]:
        """Returns a tuple of the left and right children of `node`

        This function takes the `Internal` type as its input, so it is
        guaranteed to always return the children.  See `as_internal` for
        converting general nodes to internal ones.
        """
    def other_child(self, parent: Internal, child: Node) -> Node:
        """Returns the child of `parent` other than `child`

        Throws an error if `child` isn't a child of `parent`.
        """
    def random_intersecting_edge(self, height: float, rng: RNG) -> Optional[int]:
        """Returns a random edge which intersects `height`

        "Intersects" here means that the edge parent is higher than `height`
        and the child is lower.  The comparisons are strict: if either node is
        exactly at `height`, the edge won't be picked.

        Returns `None` if there is no such node.
        """
    def edge_index(self, child: Node) -> int:
        """Returns the index of an edge from `child` to its parent"""
    def edge_length(self, edge: int) -> float:
        """Returns the length of `edge`

        The length is the distance between the parent and the child nodes of
        that edge.
        """
    def edge_nodes(self, edge: int) -> tuple[Node, Internal]:
        """Returns the `(child, parent)` tuple corresponding to an edge"""
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
    def random_nonroot_node(self, rng: RNG) -> tuple[Internal, Internal]:
        """Returns a random non-root node and its parent"""
    def random_internal(self, rng: RNG) -> Internal:
        """Returns a random internal node"""
    def random_nonroot_internal(self, rng: RNG) -> tuple[Internal, Internal]:
        """Returns a random non-root internal and its parent"""
    def random_leaf(self, rng: RNG) -> Leaf:
        """Returns a random leaf node"""
    def nodes(self) -> Iterator[Node]:
        """An iterator over all of trees nodes

        All of the `Leaf` nodes go before `Internal` ones.
        """
    def leaf_by_name(self, name: str) -> Optional[Leaf]:
        """Gets a named leaf or `None` if the name is not found"""
    def internals(self) -> Iterator[Internal]:
        """An iterator over all of the trees internal nodes"""
    def leaves(self) -> Iterator[Leaf]:
        """An iterator over all of the trees leaf nodes"""
    def total_length(self) -> float:
        """The total length of all tree edges"""
    def has_dated_tips(self) -> bool:
        """Returns `True` if any of the leaves have non-0 height"""
    def validate(self) -> None:
        """Throws an exception if a tree is malformed

        This function ensures that:

        - No leaf has become anyone's parent.
        - All parent nodes are older than their children.
        - Parents match their children (mismatches can happen when
          `update_edge` is used incorrectly).
        - There's only one root (two or more can be set with `set_root`).
        - The tree is a tree, meaning that topologically it has no cycles and
          is connected.
        """
    def newick(self, internal_ids=False) -> str:
        """Returns the tree topology in the Newick format

        Leaf nodes will be labeled with the names passed to the constructor
        while the internal nodes are unlabeled.
        """

class Leaf(Hashable):
    """Leaf node of the phylogenetic tree

    Leaf nodes are the ones which are associated with a concrete sequence.
    Currently all leaf nodes have the distance of $0$, although that'll be
    subject to change in the future.
    """

class Internal(Hashable):
    """Internal anonymous node of the phylogenetic tree.

    Internals are the unnamed ancestors which form the tree.
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

class CPU4Likelihood(Likelihood):
    def __init__(
        self,
        msa: MSA,
        substitution: JC | K80 | HKY,
        # TODO: types
        clock: Any,
        tree: Tree,
    ): ...

class Thread4Likelihood(Likelihood):
    def __init__(
        self,
        msa: MSA,
        substitution: JC | K80 | HKY,
        # TODO: types
        clock: Any,
        tree: Tree,
        *,
        thread_split_size: int = 400,
    ): ...

class CUDALikelihood(Likelihood):
    def __init__(
        self,
        msa: MSA,
        substitution: JC | K80 | HKY,
        # TODO: types
        clock: Any,
        tree: Tree,
        *,
        cuda_device: int = 0,
    ): ...

class MCMC:
    """
    The main object which runs the analysis
    """

    def __init__(
        self,
        burnin: int,
        length: int,
        state: Sequence[Stateful],
        priors: Sequence[Prior],
        operators: Sequence[Operator],
        likelihood: Likelihood,
        callbacks: Sequence[Callback],
        rng: RNG,
    ): ...
    @property
    def current_step(self) -> int:
        """
        Index of the current MCMC step

        Starts from 0, includes burn-in.
        """
    @property
    def state(self) -> list[Stateful]:
        """
        All stateful objects tracked by this `MCMC` instance
        """
    @property
    def priors(self) -> list[Prior]:
        """
        All priors
        """
    @property
    def operators(self) -> list[Operator]:
        """
        A list of all operator objects
        """
    @property
    def likelihood(self) -> Likelihood: ...
    @property
    def callbacks(self) -> list[Callback]:
        """
        All active callbacks
        """
    @property
    def rng(self) -> RNG:
        """
        Randomness source of this analysis

        This objects is passed to operators and used for internal randomness
        generation (such as picking the operator on each step).  Since the
        underlying object is shared, using it will alter the rest of the
        analysis.
        """
    @property
    def posterior(self) -> float:
        """Posterior probability for the last accepted step"""

    @property
    def prior(self) -> float:
        """Prior likelihood for the current step

        Note that unlike [`posterior`](#MCMC.posterior) and
        [`Likelihood`](#MCMC.Likelihood), this property isn't cached.  It will
        trigger a recalculation on all priors on each access.
        """

    @property
    def operator_statistics(
        self,
    ) -> list[tuple[Operator, list[int], timedelta, timedelta]]:
        """
        Operator statistics for this run

        Returns a list of `(operator, results, propose, likelihood)` tuples for
        each operator.  `propose` and `likelihood` records the total time the
        MCMC spent waiting for the operator to generate a proposal and
        calculate it respectively.  `operator` is the reference to the original
        operator object.  And `results` is a list of step results.
        """

    def run(self) -> None:
        """Start the simulation

        This yields flow control to the Rust core until the simulation is done.
        Press Ctrl+C to interrupt and stop the execution.
        """

    def measure_operator(self, operator_index: int, length: int) -> list[int]:
        """TODO"""

@dataclass
class EpochScale(Operator):
    """Scales a random epoch in a tree

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the height of 0).
    """

    tree: Tree
    factor: float
    """
    The scaling ratio will be sampled from `(factor, 1 / factor)`.  So, the
    factor must be between 0 and 1 and the smaller it is the larger the steps
    will be.
    """
    distribution: Distribution
    """Distribution from which the scale is sampled."""
    rng: RNG
    weight: float = 1

@dataclass(slots=True)
class SubtreeLeap(Operator):
    """
    Moves a node a distance, changing the topology randomly

    First, a distance delta is sampled from the distribution.  The operator
    selects a random node and all edges `delta` away from that node (down if
    the delta is negative or up and down if it's positive).  One of those edges
    is randomly selected and the node is spliced into it.  If the delta is
    above the root, the node will become the new root.
    """

    tree: Tree
    """The tree to edit."""
    distribution: Sample[float]
    """The distribution to draw the height move distance from"""
    rng: RNG
    weight: float = 1

@dataclass
class ConstantPopulation(Prior):
    """Constant population coalescent"""

    tree: Tree
    population: SupportsFloat

@dataclass
class ExponentialGrowth(Prior):
    tree: Tree
    population: SupportsFloat
    growth_rate: SupportsFloat

@dataclass
class Monophyly(Prior):
    """
    Ensures that a group of leaves form a monophyly

    Returns a static probability if the specified leaves are monophyletic or
    aborts the move otherwise.
    """

    tree: Tree
    leaves: Sequence[Leaf]

class Yule(Prior):
    """Uncalibrated Yule birth-rate model"""

    tree: Tree
    birth_rate: SupportsFloat

class Real(Stateful, Scalable):
    """
    Real multidimensional parameter

    It acts as a list of `float`s, except it implements
    [`Stateful`](#Stateful).  Note that this uses regular 64-bit floating point
    numbers underneath, not some custom arbitrary precision real type.
    """
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
    """
    Integer multidimensional parameter

    It acts as a list of `int`s, except it implements [`Stateful`](#Stateful).
    """
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
    """
    Boolean multidimensional parameter

    It acts as a list of `bool`s, except it implements [`Stateful`](#Stateful).
    """

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
