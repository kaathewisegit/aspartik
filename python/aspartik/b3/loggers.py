"""Classes which record the state of the simulation.

All classes here adhere to the `Callback` protocol and should be passed as
such.
"""

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import TextIOBase
from typing import Any, Optional

from . import MCMC, Callback, Prior, Tree
from .parameters import Real, Weights


@dataclass(slots=True)
class TreeLogger(Callback):
    """Records the topology of the tree into a `.trees` file."""

    tree: Tree
    path: str
    """
    Path to the file where the trees will be appended in Newick format, one per
    line.  It's opened verbatim (the `.trees` extension won't be added).
    """
    every: int
    """How often the logger will be called"""

    def __post_init__(self):
        self._file = open(self.path, "w")

    def log(self, mcmc: MCMC):
        line = self.tree.newick()
        self._file.write(line)
        self._file.write("\n")

    def __getstate__(self):
        # None ignores __dict__ which contains the file handle
        return (None, self.__slots__)


@dataclass(slots=True)
class PrintLogger(Callback):
    every: int

    _last_time: Optional[float] = field(init=False, default=None)

    def log(self, mcmc: MCMC):
        if mcmc.current_step == 0:
            print(
                f"{'Step':>10}{'Posterior':>15}{'Likelihood':>15}{'Prior':>15}{'Speed t/m':>15}"
            )

        current_time = time.perf_counter()
        if self._last_time:
            # in seconds
            speed = (current_time - self._last_time) / self.every * 1_000_000
            if speed >= 60:
                speed = f"{speed / 60:.1f}min"
            else:
                speed = f"{speed:.1f}sec"
        else:
            speed = "-"

        likelihood = mcmc.likelihood.likelihood()
        print(
            f"{mcmc.current_step:>10}{mcmc.posterior:>15.2f}{likelihood:>15.2f}{mcmc.prior:>15.2f}{speed:>15}"
        )

        self._last_time = current_time


def _serialize(item):
    if isinstance(item, Real):
        return float(item)

    elif isinstance(item, Weights):
        return list(item)

    elif isinstance(item, Prior):
        return item.probability()

    elif callable(item):
        return item()


@dataclass(slots=True)
class ValueLogger(Callback):
    items: Mapping[str, Any]
    path: str
    every: int

    _file: TextIOBase = field(init=False)

    def __post_init__(self):
        self._file = open(self.path, "w")

    def log(self, mcmc: MCMC):
        entry_json = json.dumps(self.items, default=_serialize)
        self._file.write(entry_json)
        self._file.write("\n")
        self._file.flush()

    def __getstate__(self):
        return (None, self.__slots__)
