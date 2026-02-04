from dataclasses import dataclass
from math import log

from ...rng import RNG
from .. import Operator, Proposal
from ..parameters import Tree


@dataclass(slots=True)
class WilsonBalding(Operator):
    """A version of a subtree regraft move

    Introduced in [this paper][paper], it picks a random subtree and inserts it
    in-between two other nodes.

    [paper]: https://doi.org/10.1093/genetics/161.3.1307
    """

    tree: Tree
    rng: RNG
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
        i_parent_height = tree.height_of(i_parent)

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

            if tree.height_of(j_parent) > i_parent_height > tree.height_of(j):
                break

        before = tree.height_of(i_grandparent) - max(
            tree.height_of(i), tree.height_of(i_brother)
        )
        after = tree.height_of(j_parent) - max(tree.height_of(i), tree.height_of(j))
        ratio = log(after / before)

        # Cut out i_parent and replace it with a direct edge from grandparent
        # to i_brother
        tree.replace_child(i_grandparent, i_parent, i_brother)
        # Hook up i_parent to j_parent.  It's fine because we checked that
        # i_parent is lower than j_parent when selecting j
        tree.replace_child(j_parent, j, i_parent)
        # Replace i_brother edge from i_parent with j.  Once again, we've
        # enforced i_parent being above j earlier
        tree.replace_child(i_parent, i_brother, j)

        return Proposal.Hastings(ratio)
