import time
from dataclasses import dataclass, field

from . import MCMC, Callback
from .parameters import Tree


@dataclass(slots=True)
class TreeValidator(Callback):
    """
    Checks that the trees it passed to it are valid

    This callback simply calls the `validate` method on all trees it tracks.
    By default it is called on every step, which will slow the analysis down
    considerably on large trees.  This callback is mostly intended for debug
    purposes when developing new operators.
    """

    trees: tuple[Tree]
    """An iterable of trees to validate"""
    every: int = 1

    def call(self, mcmc: MCMC) -> None:
        for tree in self.trees:
            tree.validate()


@dataclass(slots=True)
class Timer(Callback):
    """
    Prints the total execution time of an MCMC run
    """

    __start: float = field(init=False, default=0.0)
    every: int = field(init=False, default=2**32)

    def call(self, mcmc: MCMC) -> None:
        "Due to high `every` this method will only be called once on step 0"

        self.__start = time.perf_counter()

    def finish(self) -> None:
        print("Timer:", time.perf_counter() - self.__start)
