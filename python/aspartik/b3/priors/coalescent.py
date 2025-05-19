from dataclasses import dataclass

from .. import Tree


@dataclass
class ConstantPopulation:
    tree: Tree

    # TODO
    def probability(self) -> float: ...


# TODO: Skyline
