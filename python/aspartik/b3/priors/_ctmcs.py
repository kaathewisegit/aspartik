from dataclasses import dataclass
from math import log

from .. import Prior, Tree
from ..parameters import Real

LN_GAMMA_1_2 = 0.5723649429247001


@dataclass(slots=True)
class CTMCS(Prior):
    """
    A rough analogue of BEAST's `ctmcsScalePrior`
    """

    tree: Tree
    parameter: Real

    def probability(self) -> float:
        out = 0.0

        # TODO: tree length normalization via the sub model
        tree_length = self.tree.total_length()
        norm = 0.5 * log(tree_length) - LN_GAMMA_1_2

        for value in self.parameter:
            out += norm - 0.5 * log(value) - value * tree_length

        return out
