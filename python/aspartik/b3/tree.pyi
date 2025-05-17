class Leaf:
    """Leaf node of the phylogenetic tree

    Leaf nodes are the ones which are associated with a concrete sequence.
    Currently all leaf nodes have the distance of $0$, although that'll be
    subject to change in the future.
    """

class Internal:
    """Internal anonymous node of the phylogenetic tree.

    Internals are the unnamed ancestors which form the tree.
    """

Node = Leaf | Internal
"""Any node of the phylogenetic tree

Used for type hints in places where there isn't a need to distinguish between
internal and leaf nodes.
"""
