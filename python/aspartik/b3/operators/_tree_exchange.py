import math

from .. import State, Tree, Proposal
from ..tree import Internal, Node


class NarrowExchange:
    def __init__(self, weight: float = 1):
        self.weight = weight

    def propose(self, state: State) -> Proposal:
        tree = state.tree
        rng = state.rng

        if tree.num_internals < 2:
            return Proposal.Reject()

        grandparent = None
        while grandparent is None:
            internal = tree.random_internal(state.rng)
            if is_grandparent(tree, internal):
                grandparent = internal

        left, right = tree.children_of(grandparent)
        if tree.weight_of(left) < tree.weight_of(right):
            parent, uncle = left, right
        elif tree.weight_of(right) < tree.weight_of(left):
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
            if is_grandparent(tree, node):
                num_grandparents_before += 1

        before = int(is_grandparent(tree, parent)) + int(is_grandparent(tree, uncle))

        if rng.random_bool(0.5):
            child = tree.children_of(parent)[0]
        else:
            child = tree.children_of(parent)[1]

        tree.swap_parents(uncle, child)

        after = int(is_grandparent(tree, parent)) + int(is_grandparent(tree, uncle))
        num_grandparents_after = num_grandparents_before - before + after
        ratio = math.log(num_grandparents_before / num_grandparents_after)

        return Proposal.Hastings(ratio)


def is_grandparent(tree: Tree, node: Internal) -> bool:
    left, right = tree.children_of(node)
    return tree.is_internal(left) and tree.is_internal(right)


class WideExchange:
    def __init__(self, weight: float = 1):
        self.weight = weight

    def propose(self, state: State) -> Proposal:
        tree = state.tree
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

        # TODO: custom `eq` implementation for node types
        if (
            j != i_parent
            and tree.weight_of(j) < tree.weight_of(i_parent)
            and tree.weight_of(i) < tree.weight_of(j_parent)
        ):
            tree.swap_parents(i, j)

            return Proposal.Hastings(0.0)
        else:
            return Proposal.Reject()
