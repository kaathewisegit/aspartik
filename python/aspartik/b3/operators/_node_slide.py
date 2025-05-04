from ._util import sample_range
from .. import State, Proposal


class InternalNodeSlide:
    """Slides a random internal node between its parent and children"""

    def __init__(
        self,
        tree,
        distribution,
        weight: float = 1,
    ):
        """
        Args:
            tree: The tree to edit.
            distribution:
                The distribution which will sample the new node height on the
                interval between its parent and the closest child.
        """
        self.tree = tree
        self.distribution = distribution
        self.weight = weight

    def propose(self, state: State) -> Proposal:
        """
        If there are no non-root internal nodes, the operator will bail with
        `Proposal.Reject`.
        """

        tree = self.tree
        rng = state.rng

        # automatically fail on trees without non-root internal nodes
        if tree.num_internals == 1:
            return Proposal.Reject()

        # Pick a non-root internal node
        node = tree.random_internal(rng)
        parent = tree.parent_of(node)
        while parent is None:
            node = tree.random_internal(rng)
            parent = tree.parent_of(node)

        left, right = tree.children_of(node)

        low = tree.weight_of(parent)
        high = min(tree.weight_of(left), tree.weight_of(right))

        new_weight = sample_range(low, high, self.distribution, rng)

        tree.update_weight(node, new_weight)

        # TODO: scale from `sample_range`
        return Proposal.Hastings(0)
