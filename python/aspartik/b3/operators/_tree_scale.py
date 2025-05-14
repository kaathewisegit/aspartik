from math import log

from ._util import sample_range
from .. import State, Proposal, Tree


class TreeScale:
    """Full tree scale operator.

    This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
    tree.  It will scale the full tree (so, for now, only its internal nodes,
    since leaves all have the weight of 0).
    """

    def __init__(self, tree: Tree, factor: float, distribution, weight: float = 1):
        """
        Args:
            tree: The tree to scale.
            factor:
                The scaling ratio will be sampled from `(factor, 1 / factor)`.
                So, the factor must be between 0 and 1 and the smaller it is
                the larger the steps will be.
            distribution: Distribution from which the scale is sampled.
        """

        self.tree = tree
        if not 0 < factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {factor}")
        self.factor = factor
        self.distribution = distribution
        self.weight = 1

    def propose(self, state: State) -> Proposal:
        tree = self.tree
        rng = state.rng

        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, rng)

        for node in tree.nodes():
            new_weight = tree.weight_of(node) * scale
            tree.update_weight(node, new_weight)

        ratio = log(scale) * (tree.num_internals - 2)
        return Proposal.Hastings(ratio)
