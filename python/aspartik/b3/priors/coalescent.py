from .. import State, Tree


class ConstantPopulation:
    def __init__(self, tree: Tree):
        self.tree = tree

    # TODO
    def probability(self, state: State) -> float: ...


# TODO: Skyline
