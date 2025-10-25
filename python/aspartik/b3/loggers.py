"""Classes which record the state of the simulation.

All classes here adhere to the `Callback` protocol and should be passed as
such.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import TextIOBase
from typing import Any

from . import MCMC, Callback, Prior, Tree
from .parameters import Parameter


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

    def __post_init__(self):
        print(f"{'step':>16}{'posterior':>16}{'likelihood':>16}{'prior':>16}")

    def log(self, mcmc: MCMC):
        print(
            f"{mcmc.current_step:>16}{mcmc.posterior:>16.2f}{mcmc.cached_likelihood:>16.2f}{mcmc.prior:>16.2f}"
        )


@dataclass(slots=True)
class ValueLogger(Callback):
    map: Mapping[str, Any]
    path: str
    every: int

    _file: TextIOBase = field(init=False)

    def __post_init__(self):
        self._file = open(self.path, "w")

    def log(self, mcmc: MCMC):
        entry = {}

        for key, item in self.map.items():
            if isinstance(item, Parameter):
                entry[key] = item[0]
            if isinstance(item, Prior):
                entry[key] = item.probability()
            if callable(item):
                entry[key] = item()

        entry_json = json.dumps(entry)
        self._file.write(entry_json)
        self._file.write("\n")
        self._file.flush()

    def __getstate__(self):
        return (None, self.__slots__)
