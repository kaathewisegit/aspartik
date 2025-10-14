from dataclasses import dataclass
from math import log

from .. import Prior, Tree
from ..parameters import Real


@dataclass(slots=True)
class CTMCS(Prior):
    """
    A rough analog of BEAST's `ctmcsScalePrior`
    """

    tree: Tree
    parameter: Real

    def probability(self) -> float:
        tree_length = self.tree.height_of(self.tree.root)
        out = 0.0

        for value in self.parameter:
            out -= log(value * tree_length)

        return out
