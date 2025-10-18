from dataclasses import dataclass
from math import log

from ...rng import RNG
from ...stats.distributions import Sample
from .. import Operator, Proposal, Tree
from ..tree import Internal, Node


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

    def propose(self) -> Proposal:
        tree = self.tree
        rng = self.rng
        root = tree.root

        node = tree.random_node(rng)
        while node == root:
            node = tree.random_node(rng)

        parent = tree.parent_of(node)
        assert parent is not None  # checked in the loop

        grandparent = tree.parent_of(parent)
        sibling = tree.other_child(parent, node)

        delta = abs(self.distribution.sample(rng))

        destinations = walk_tree(tree, node, sibling, parent, delta)

        random_idx = rng.random_int(0, len(destinations))
        destination = list(destinations.keys())[random_idx]
        destination_parent = tree.parent_of(destination)

        if parent == destination or parent == destination_parent:
            pass
        else:
            parent_to_sibling_edge = tree.edge_index(sibling)

            if grandparent is None:
                tree.set_root(sibling)
            else:
                # grandparent -> sibling
                grandparent_to_parent = tree.edge_index(parent)
                tree.update_edge(grandparent_to_parent, sibling)

            if destination_parent is None:
                tree.update_edge(parent_to_sibling_edge, destination)
                tree.set_root(parent)
            else:
                destination_parent_to_destination = tree.edge_index(destination)
                tree.update_edge(destination_parent_to_destination, parent)

                tree.update_edge(parent_to_sibling_edge, destination)

        new_height = destinations[destination]
        tree.set_height(parent, new_height)

        reverse_destinations = walk_tree(
            tree, node, tree.other_child(parent, node), parent, delta
        )

        ratio = log(len(destinations)) - log(len(reverse_destinations))
        return Proposal.Hastings(ratio)


def walk_tree(
    tree: Tree, node: Node, sibling: Node, parent: Internal, delta: float
) -> dict[Node, float]:
    """
    Finds all of the nodes whose parent edge has a point which is `delta` away
    from `parent` and who are above `node`.
    """

    destinations = {}

    node_height = tree.height_of(node)
    parent_height = tree.height_of(parent)
    below, above = parent_height - delta, parent_height + delta

    if node_height < below:
        # if we move the parent to `below`, it'll still be above `node`, so we
        # can search for intersections with siblings
        intersections(destinations, tree, sibling, below)

    up_node = parent
    while True:
        up_parent = tree.parent_of(up_node)

        if up_parent is None:
            # up_node is root, terminate
            destinations[up_node] = above
            break

        up_parent_height = tree.height_of(up_parent)

        if up_parent_height > above:
            # up_parent is above the line, `up_node` is a valid destination
            destinations[up_node] = above
            break

        # up_node is closer than delta

        # new distance which accounts for our climb
        new_below = up_parent_height - (above - up_parent_height)

        up_sibling = tree.other_child(up_parent, up_node)

        if node_height < new_below:
            intersections(destinations, tree, up_sibling, new_below)

        up_node = up_parent

    return destinations


def intersections(
    destinations: dict[Node, float], tree: Tree, node: Node, height: float
) -> None:
    if tree.height_of(node) < height:
        destinations[node] = height
    elif isinstance(node, Internal):
        left, right = tree.children_of(node)
        intersections(destinations, tree, left, height)
        intersections(destinations, tree, right, height)
