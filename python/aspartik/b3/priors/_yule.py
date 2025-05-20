from dataclasses import dataclass
from math import log

from .. import Tree, Parameter


@dataclass
class Yule:
    tree: Tree
    birth_rate: Parameter

    # TODO: check leaf dates

    def probability(self) -> float:
        tree = self.tree

        rate = self.birth_rate[0]
        assert isinstance(rate, float)

        out = (tree.num_leaves - 1) * log(rate)

        root = tree.root()
        for node in tree.internals():
            diff = -log(rate) * tree.weight_of(node)

            out += diff
            if node == root:
                out += diff

        return out
