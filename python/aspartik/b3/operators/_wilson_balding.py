from math import log

from .. import Proposal, Tree
from ...rng import Rng


class WilsonBalding:
    """TODO."""

    def __init__(self, tree: Tree, rng: Rng, weight: float = 1):
        self.tree = tree
        self.rng = rng
        self.weight = weight

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng

        # pick a random non-root node
        while True:
            i_parent = tree.random_internal(rng)
            i_grandparent = tree.parent_of(i_parent)
            if i_grandparent is not None:
                break

        i, i_brother = tree.children_of(i_parent)
        if rng.random_bool():
            i, i_brother = i_brother, i

        # Pick a node j_parent, such that it's above i_parent and one of its
        # children is below i_parent
        while True:
            j_parent = tree.random_internal(rng)
            j, j_brother = tree.children_of(j_parent)
            if rng.random_bool():
                j, j_brother = j_brother, j

            if tree.weight_of(j_parent) > tree.weight_of(i_parent) > tree.weight_of(j):
                break

        before = tree.weight_of(i_grandparent) - max(
            tree.weight_of(i), tree.weight_of(i_brother)
        )
        after = tree.weight_of(j_parent) - max(tree.weight_of(i), tree.weight_of(j))
        ratio = log(after / before)

        # Cut out i_parent and replace it with a direct edge from grandparent
        # to i_brother
        tree.update_edge(tree.edge_index(i_parent), i_brother)
        # Hook up i_parent to j_parent.  It's fine because we checked that
        # i_parent is lower than j_parent when selecting j
        tree.update_edge(tree.edge_index(j), i_parent)
        # Replace i_brother edge from i_parent with j.  Once again, we've
        # enforced i_parent being above j earlier
        tree.update_edge(tree.edge_index(i_brother), j)

        return Proposal.Hastings(ratio)
