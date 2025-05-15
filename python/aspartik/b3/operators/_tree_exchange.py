import math

from .. import State, Tree, Proposal
from ..tree import Internal, Node


class NarrowExchange:
    """Operator which exchanges the parents of two neighbouring nodes.

    This operator is analogous to BEAST2's `Exchange` operator with `isNarrow`
    set to true.  It finds a grandparent (internal node both of whose children
    are also internal) with two kids: *parent* and *uncle* (uncle is younger
    than the parent).  And one of the children of *parent* is swapped with
    *uncle*.
    """

    def __init__(self, tree: Tree, weight: float = 1):
        self.tree = tree
        self.weight = weight

    def propose(self, state: State) -> Proposal:
        tree = self.tree
        rng = state.rng

        if tree.num_internals < 2:
            return Proposal.Reject()

        grandparent = None
        while grandparent is None:
            internal = tree.random_internal(state.rng)
            if tree.is_grandparent(internal):
                grandparent = internal

        left, right = tree.children_of(grandparent)
        if tree.weight_of(left) > tree.weight_of(right):
            parent, uncle = left, right
        elif tree.weight_of(right) > tree.weight_of(left):
            parent, uncle = right, left
        else:
            return Proposal.Reject()

        parent, uncle = tree.as_internal(parent), tree.as_internal(uncle)
        # If the lower child isn't internal, abort.
        if parent is None:
            return Proposal.Reject()
        assert isinstance(parent, Internal)
        assert isinstance(uncle, Internal)

        num_grandparents_before = 0
        for node in tree.internals():
            if tree.is_grandparent(node):
                num_grandparents_before += 1

        before = int(tree.is_grandparent(parent)) + int(tree.is_grandparent(uncle))

        if rng.random_bool(0.5):
            child = tree.children_of(parent)[0]
        else:
            child = tree.children_of(parent)[1]

        tree.swap_parents(uncle, child)

        after = int(tree.is_grandparent(parent)) + int(tree.is_grandparent(uncle))
        num_grandparents_after = num_grandparents_before - before + after
        ratio = math.log(num_grandparents_before / num_grandparents_after)

        return Proposal.Hastings(ratio)


class WideExchange:
    """Operator which exchanges the parent of two random nodes.

    This operator is analogous to BEAST2's `Exchange` operator with `isNarrow`
    set to false.  It picks two random nodes in the tree (they could be either
    leaves or internals) and swaps their parents.

    If a randomly selected move is impossible (a parent would be younger than
    its child) the operator aborts with `Proposal.Reject`.
    """

    def __init__(self, tree: Tree, weight: float = 1):
        self.tree = tree
        self.weight = weight

    def propose(self, state: State) -> Proposal:
        tree = self.tree
        rng = state.rng

        i = tree.random_node(rng)
        j = None
        while j is None or j != i:
            j = tree.random_node(rng)
        assert isinstance(j, Node)

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
            and tree.weight_of(j) < tree.weight_of(i_parent)
            and tree.weight_of(i) < tree.weight_of(j_parent)
        ):
            tree.swap_parents(i, j)

            return Proposal.Hastings(0.0)
        else:
            return Proposal.Reject()
