from math import log
from dataclasses import dataclass

from .. import Proposal, Tree
from ...rng import Rng


@dataclass
class WilsonBalding:
    """TODO."""

    tree: Tree
    rng: Rng
    weight: float = 1

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
                j = j_brother

            if tree.weight_of(j_parent) > tree.weight_of(i_parent) > tree.weight_of(j):
                break

        before = tree.weight_of(i_grandparent) - max(
            tree.weight_of(i), tree.weight_of(i_brother)
        )
        after = tree.weight_of(j_parent) - max(tree.weight_of(i), tree.weight_of(j))
        ratio = log(after / before)

        i_grandparent_to_i_parent = tree.edge_index(i_parent)
        j_parent_to_j = tree.edge_index(j)
        i_parent_to_i_brother = tree.edge_index(i_brother)

        # Cut out i_parent and replace it with a direct edge from grandparent
        # to i_brother
        tree.update_edge(i_grandparent_to_i_parent, i_brother)
        # Hook up i_parent to j_parent.  It's fine because we checked that
        # i_parent is lower than j_parent when selecting j
        tree.update_edge(j_parent_to_j, i_parent)
        # Replace i_brother edge from i_parent with j.  Once again, we've
        # enforced i_parent being above j earlier
        tree.update_edge(i_parent_to_i_brother, j)

        return Proposal.Hastings(ratio)
