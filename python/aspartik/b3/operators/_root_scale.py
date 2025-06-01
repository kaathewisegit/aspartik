from math import log

from ._util import sample_range
from . import TreeScale
from .. import Proposal, Operator


class RootScale(TreeScale, Operator):
    """Scales the root node

    This parameter has the same functionality as `TreeScale`, except it only
    scales the root node and not all internals.

    This operator will reject if the root node gets moved lower than its
    children.
    """

    def propose(self) -> Proposal:
        tree = self.tree

        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        root = tree.root()
        new_weight = tree.weight_of(root) * scale
        left, right = tree.children_of(root)

        if tree.weight_of(left) >= new_weight or tree.weight_of(right) >= new_weight:
            return Proposal.Reject()
        else:
            tree.update_weight(root, tree.weight_of(root) * scale)

        ratio = log(scale)
        return Proposal.Hastings(ratio)
