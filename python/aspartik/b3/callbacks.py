import time
from dataclasses import dataclass, field
from typing import Optional

from .._aspartik_rust_impl._b3_rust_impl import (
    OperatorStats as OperatorStats,
    TraceWriter as TraceWriter,
)
from . import MCMC, Callback
from .parameters import Tree


@dataclass(slots=True)
class PrintLogger(Callback):
    """
    Prints the simulation progress onto the screen

    Currently it only supports the step index, posterior/likelihood/total
    prior, and speed in time per million steps.
    """

    every: int

    _last_time: Optional[float] = field(init=False, default=None)

    def call(self, mcmc: MCMC):
        if mcmc.current_step == 0:
            print(
                f"{'Step':>10}{'Posterior':>15}{'Likelihood':>15}{'Prior':>15}{'Speed t/m':>15}"
            )

        current_time = time.perf_counter()
        if self._last_time:
            # in seconds
            speed = (current_time - self._last_time) / self.every * 1_000_000

            speed = f"{speed / 60:.1f}min" if speed >= 60 else f"{speed:.0f}sec"
        else:
            speed = "-"

        likelihood = mcmc.likelihood.likelihood()
        print(
            f"{mcmc.current_step:>10}{mcmc.posterior:>15.2f}{likelihood:>15.2f}{mcmc.prior:>15.2f}{speed:>15}"
        )

        self._last_time = current_time


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
        if not self.__start:
            return

        duration = time.perf_counter() - self.__start
        speed = duration / (mcmc.current_step - self.__start_index) * 1_000_000
        print(f"Timer: {speed} sec/million steps ({duration} sec total)")


@dataclass(slots=True)
class StateCheckpoint(Callback):
    """
    Saves the MCMC state
    """

    path: str
    "Path to the file to save the state in"

    every: int

    def save_state(self, mcmc: MCMC):
        with open(self.path, "wb") as file:
            file.write(mcmc.dump_state())

    def call(self, mcmc: MCMC) -> None:
        self.save_state(mcmc)

    def finish(self, mcmc: MCMC) -> None:
        self.save_state(mcmc)
