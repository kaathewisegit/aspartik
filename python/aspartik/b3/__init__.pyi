from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Any, ClassVar, Literal, Optional, Protocol, Tuple, runtime_checkable

from ..data import DNASeq
from ..data.msa import MSA
from ..data.newick import Tree as NewickTree
from ..rng import RNG
from .substitutions import HKY, JC, K80
from .tree import Internal, Leaf, Node

class tree: ...

@runtime_checkable
class Stateful(Protocol):
    """
    Epoch-versioned objects for use with MCMC

    All stateful objects handled by `MCMC` must conform to this protocol.
    During each step operators can edit objects using whatever APIs provided.
    Then, at the end of the step, `MCMC` calls `accept` if the move has been
    accepted, or `reject` otherwise.
    """

    def accept(self) -> None:
        """Accept changes made during the current step"""

    def reject(self) -> None:
        """Reject changes made during the current step

        This method must roll the state of the object back to how it was at the
        beginning of the MCMC step.
        """

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

        All heights are initially set to 0.  Use `set_leaf_heights` for dated
        tips and `set_random_heights` to randomize the positions of internal
        nodes.
        """

    @classmethod
    def from_newick(_cls, newick: NewickTree) -> Tree:
        """
        Initializes the tree from a Newick object.

        The Newick tree must be strictly bifurcating and all of its edges must
        have a defined length.
        """

    def set_random_topology(self, rng: RNG):
        """TODO"""

    def set_leaf_heights(self, heights: Sequence[float]):
        """
        Set the tip dates.

        When a tree is first created, all of the nodes have a height of 0.
        This method allows to date leaf nodes.  The length of `heights` must be
        equal `num_leaves`.
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
    def random_nonroot_node(self, rng: RNG) -> Tuple[Internal, Internal]:
        """Returns a random non-root node and its parent"""
    def random_internal(self, rng: RNG) -> Internal:
        """Returns a random internal node"""
    def random_nonroot_internal(self, rng: RNG) -> Tuple[Internal, Internal]:
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

class Likelihood:
    """
    Tree likelihood calculator

    This object calculates the likelihood of a tree given the sequence data
    using Felsenstein's tree pruning algorithm.

    There are several implementations, each with its own options (**TODO**
    docs).
    """

    def __init__(
        self,
        msa: MSA,
        substitution: JC | K80 | HKY,
        # TODO: types
        clock: Any,
        tree: Tree,
        calculator: Literal["cpu", "thread", "cuda"] = "cpu",
        cuda_device: int = 0,
        thread_split_size: int = 400,
    ): ...

@runtime_checkable
class Prior(Protocol):
    """
    Interface which describes all prior distributions

    A `Prior` object will be queried by `MCMC` on each step after the operator
    edits the state to get the prior probability of the new state.  It's the
    responsibility of the prior to track the stateful objects it's interested
    in, so they will typically be passed in the constructor.

    Variable priors, coalescents, birth-death models, and all other
    non-likelihood models implement this protocol.
    """

    def probability(self) -> float:
        """Calculates the log prior probability of the model state

        The return value must be a **natural logarithm** of the probability.

        `MCMC` will short-circuit and abort the move if `probability` returns a
        negative infinity.  This can be used to avoid expensive likelihood
        calculations for obviously invalid moves, like going out of variable
        bounds.
        """

class Operator(Protocol):
    """
    Objects which propose moves by editing state

    It is the responsibility of the objects to track the parts of the state it
    might want to edit.  Typically these objects will be passed in the
    constructor.
    """

    def propose(self) -> Proposal:
        """Proposes a new MCMC step

        It is presumed that the operator will store all the references to
        parameters and trees it wants to edit and will change them accordingly.
        If a move cannot be proposed for any reason `Proposal.Reject` should be
        returned.  `MCMC` will deal with rolling back the state.
        """

    @property
    def weigth(self) -> float:
        """Influences the probability of the operator being picked

        On each step `MCMC` picks a random operator from the list passed to it.
        It uses this value to weight them.  So, the larger it is, the more
        often the operator will be picked, and visa versa.  This value is read
        once on startup.  Therefore, if it's changed mid-execution the old
        cached value will still be used.
        """

class Callback(Protocol):
    """
    Custom callbacks

    `b3` supports arbitrary logging and checks via this protocol.  The `call`
    function is passed a reference to the main `MCMC` object, so the callback
    can either take state variables in the constructor of fetch them via the
    `MCMC` attributes.

    The `call` function won't be called after each step for efficiency.  See
    [`every`](#Callback.every) for configuring how often the callback will be
    invoked.
    """

    every: int
    """How often this callback should be called

    The `MCMC` will call each callback object when `index % every` is 0.  This
    value is read once when MCMC is created, so if it's changed during
    execution, the old `every` value will continue to be used.
    """

    def call(self, mcmc: MCMC) -> None:
        """
        A custom operation

        Used by loggers and other periodic actions.
        """

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
        likelihoods: Sequence[Likelihood],
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
    def likelihoods(self) -> list[Likelihood]:
        """
        All accounted for likelihoods
        """
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
    def cached_likelihood(self) -> float:
        """Total likelihood for the last accepted step"""

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
