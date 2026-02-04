from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Sample
from .. import Operator, Proposal
from ..parameters import Internal, Node, Tree
from ._util import family


@dataclass(slots=True)
class SubtreeSlide(Operator):
    """"""

    tree: Tree
    """The tree to edit."""
    distribution: Sample[float]
    """
    The distribution which will sample the new node height on the interval
    between its parent and the closest child.
    """
    rng: RNG
    weight: float = 1

    def propose(self) -> Proposal:
        """
        If there are no non-root internal nodes, the operator will bail with
        `Proposal.Reject`.
        """

        rng = self.rng
        tree = self.tree

        ratio = 0.0

        # automatically fail on trees without non-root internal nodes
        if tree.num_internals == 1:
            return Proposal.Reject()

        node, sibling, parent, grandparent = family(tree, rng)
        delta = self.distribution.sample(rng)
        old_height = tree.height_of(parent)
        new_height = old_height + delta

        if delta > 0:
            if grandparent is not None and tree.height_of(grandparent) < new_height:
                # two nodes whose edge intersects `new_height`
                new_child = parent
                new_parent = grandparent

                while tree.height_of(new_parent) < new_height:
                    new_child = new_parent
                    new_parent = tree.parent_of(new_child)
                    if new_parent is None:
                        break

                if new_parent is None:
                    # new_child was the root
                    tree.replace_child(parent, sibling, new_child)
                    tree.replace_child(grandparent, parent, sibling)
                    tree.set_root(parent)
                else:
                    tree.replace_child(parent, sibling, new_child)
                    tree.replace_child(grandparent, parent, sibling)
                    tree.replace_child(new_parent, new_child, parent)

                num_reverse_sources = len(intersections(tree, new_child, old_height))
                ratio = -log(num_reverse_sources)
        else:
            if tree.height_of(node) > new_height:
                return Proposal.Reject()

            if tree.height_of(sibling) > new_height:
                # topological changes

                destinations = intersections(tree, sibling, new_height)
                random_idx = rng.random_int(0, len(destinations))
                new_child = destinations[random_idx]
                new_parent = tree.parent_of(new_child)
                assert new_parent is not None

                if grandparent is None:
                    tree.replace_child(parent, sibling, new_child)
                    tree.replace_child(new_parent, new_child, parent)

                    tree.set_root(sibling)
                else:
                    tree.replace_child(parent, sibling, new_child)
                    tree.replace_child(grandparent, parent, sibling)
                    tree.replace_child(new_parent, new_child, parent)

                ratio = -log(len(destinations))

        tree.set_height(parent, new_height)

        return Proposal.Hastings(ratio)


def intersections(tree: Tree, node: Node, height: float) -> list[Node]:
    if tree.height_of(node) < height:
        return [node]

    if isinstance(node, Internal):
        left, right = tree.children_of(node)
        return intersections(tree, left, height) + intersections(tree, right, height)
    else:
        return []
