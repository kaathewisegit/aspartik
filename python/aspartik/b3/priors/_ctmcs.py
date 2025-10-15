from dataclasses import dataclass
from math import log

from .. import Prior, Tree
from ..parameters import Real


@dataclass(slots=True)
class CTMCS(Prior):
    """
    A rough analogue of BEAST's `ctmcsScalePrior`
    """

    tree: Tree
    parameter: Real

    def probability(self) -> float:
        out = 0.0

        tree_length = self.tree.total_length()
        # TODO: tree length normalization via the sub model
        norm = 0.5 * log(tree_length)

        for value in self.parameter:
            out += norm - 0.5 * log(value) - value * tree_length

        return out
