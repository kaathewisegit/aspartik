import time
from dataclasses import dataclass, field

from .._aspartik_rust_impl._b3_rust_impl import (
    TraceWriter as TraceWriter,
)
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
    __start_index: int = field(init=False)
    every: int = field(init=False, default=2**32)

    def call(self, mcmc: MCMC) -> None:
        "Due to high `every` this method will only be called once on step 0"

        if self.__start == 0.0:
            self.__start = time.perf_counter()
            self.__start_index = mcmc.current_step

    def finish(self, mcmc: MCMC) -> None:
        duration = time.perf_counter() - self.__start
        speed = duration / (mcmc.current_step - self.__start_index) * 1_000_000
        print(f"Timer: {speed} sec/million steps ({duration} sec total)")
