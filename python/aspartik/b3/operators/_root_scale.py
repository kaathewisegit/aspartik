from math import log

from ._util import sample_range
from . import TreeScale
from .. import Proposal


class RootScale(TreeScale):
    """Scales the root node.

    This parameter has the same functionality as `TreeScale`, except it only
    scales the root node and not all internals.
    """

    def propose(self) -> Proposal:
        tree = self.tree

        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        root = tree.root()
        tree.update_weight(root, tree.weight_of(root) * scale)

        ratio = log(scale)
        return Proposal.Hastings(ratio)
