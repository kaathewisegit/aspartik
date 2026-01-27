from dataclasses import dataclass
from math import log

from ...rng import RNG
from .. import Operator, Proposal
from ..parameters import Internal, Node, Tree
from ._util import assert_two_internals


@dataclass(slots=True)
class NarrowExchange(Operator):
    """Exchanges the parents of two neighbouring nodes

    This operator is analogous to BEAST2's `Exchange` operator with `isNarrow`
    set to true.  It finds a grandparent (internal node both of whose children
    are also internal) with two kids: `parent` and `uncle` (uncle is younger
    than the parent).  And one of the children of `parent` is swapped with
    `uncle`.
    """

    tree: Tree
    rng: RNG
    weight: float = 1

    def __post_init__(self):
        assert_two_internals(self)

    def propose(self) -> Proposal:
        tree = self.tree

        num_grandparents_before = tree.num_grandparents()
        if num_grandparents_before == 0:
            # no grandparents to pick `grandparent` from
            return Proposal.Reject()

        while True:
            grandparent = tree.random_internal(self.rng)
            if tree.is_grandparent(grandparent):
                break

        left, right = tree.children_of(grandparent)
        if tree.height_of(left) > tree.height_of(right):
            parent, uncle = left, right
        elif tree.height_of(right) > tree.height_of(left):
            parent, uncle = right, left
        else:
            return Proposal.Reject()

        # guaranteed because `grandparent` is a grandparent
        assert isinstance(parent, Internal)
        assert isinstance(uncle, Internal)

        before = int(tree.is_grandparent(parent)) + int(tree.is_grandparent(uncle))

        if self.rng.random_bool(0.5):
            child = tree.children_of(parent)[0]
        else:
            child = tree.children_of(parent)[1]

        tree.swap_parents(uncle, child)

        after = int(tree.is_grandparent(parent)) + int(tree.is_grandparent(uncle))
        num_grandparents_after = num_grandparents_before - before + after
        ratio = log(num_grandparents_before / num_grandparents_after)

        return Proposal.Hastings(ratio)


@dataclass(slots=True)
class WideExchange(Operator):
    """Exchanges the parent of two random nodes

    This operator is analogous to BEAST2's `Exchange` operator with `isNarrow`
    set to false.  It picks two random nodes in the tree (they could be either
    leaves or internals) and swaps their parents.

    If a randomly selected move is impossible (a parent would be younger than
    its child) the operator aborts with `Proposal.Reject`.
    """

    tree: Tree
    rng: RNG
    weight: float = 1

    def propose(self) -> Proposal:
        tree = self.tree

        root = tree.root

        i = tree.random_node(self.rng)
        while i == root:
            i = tree.random_node(self.rng)

        j = None
        while j is None or j == i or j == root:
            j = tree.random_node(self.rng)
        assert j is not None

        i_parent = tree.parent_of(i)
        if i_parent is None:
            return Proposal.Reject()
        j_parent = tree.parent_of(j)
        if j_parent is None:
            return Proposal.Reject()

        # Abort if j and i are parent-child or if one of the parents would be
        # younger than its new child or if the two selected nodes.
        if (
            j != i_parent
            and i != j_parent
            and tree.height_of(j) < tree.height_of(i_parent)
            and tree.height_of(i) < tree.height_of(j_parent)
        ):
            tree.swap_parents(i, j)

            return Proposal.Hastings(0.0)
        else:
            return Proposal.Reject()


@dataclass(slots=True)
class BeastNarrowExchange(Operator):
    """
    Narrow exchange operator compatible with BEASTs `narrowExchange`
    """

    tree: Tree
    rng: RNG
    weight: float = 1

    def propose(self):
        rng = self.rng
        tree = self.tree
        root = tree.root

        node = tree.random_node(rng)
        while node == root or tree.parent_of(node) == root:
            node = tree.random_node(rng)

        parent = tree.parent_of(node)
        assert parent is not None
        grandparent = tree.parent_of(parent)
        assert grandparent is not None
        uncle = tree.other_child(grandparent, parent)

        if tree.height_of(uncle) < tree.height_of(parent):
            tree.swap_parents(node, uncle)
            return Proposal.Hastings(0.0)
        else:
            return Proposal.Reject()


@dataclass(slots=True)
class BeastWideExchange(Operator):
    """
    Wide exchange operator compatible with BEASTs `wideExchange`
    """

    tree: Tree
    rng: RNG
    weight: float = 1

    def propose(self):
        rng = self.rng
        tree = self.tree
        root = tree.root

        node_a = tree.random_node(rng)
        while node_a == root:
            node_a = tree.random_node(rng)

        node_b = tree.random_node(rng)
        while node_b == root or node_b == node_a:
            node_b = tree.random_node(rng)

        node_a_parent = tree.parent_of(node_a)
        node_b_parent = tree.parent_of(node_b)
        assert node_a_parent is not None
        assert node_b_parent is not None

        if (
            node_a_parent != node_b_parent
            and node_a != node_b_parent
            and node_b != node_a_parent
            and tree.height_of(node_a) < tree.height_of(node_b_parent)
            and tree.height_of(node_b) < tree.height_of(node_a_parent)
        ):
            tree.swap_parents(node_a, node_b)
            return Proposal.Hastings(0.0)
        else:
            return Proposal.Reject()
