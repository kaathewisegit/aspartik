"""Classes which record the state of the simulation.

All classes here adhere to the `Logger` protocol and can be passed to the `run`
function.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import TextIOBase
from typing import Any

from . import MCMC, Logger, Parameter, Prior, Tree


@dataclass(slots=True)
class TreeLogger(Logger):
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
class PrintLogger(Logger):
    every: int

    def __post_init__(self):
        print(f"{'step':>16}{'posterior':>16}{'likelihood':>16}{'prior':>16}")

    def log(self, mcmc: MCMC):
        print(
            f"{mcmc.current_step:>16}{mcmc.posterior:>16.2f}{mcmc.likelihood:>16.2f}{mcmc.prior:>16.2f}"
        )


@dataclass(slots=True)
class ValueLogger(Logger):
    map: Mapping[str, Any]
    path: str
    every: int

    _params: dict[str, Parameter] = field(default_factory=dict, init=False)
    _priors: dict[str, Prior] = field(default_factory=dict, init=False)
    _functions: dict[str, Callable] = field(default_factory=dict, init=False)
    _file: TextIOBase = field(init=False)

    def __post_init__(self):
        self._file = open(self.path, "w")

        for key, item in self.map.items():
            if isinstance(item, Parameter):
                self._params[key] = item
            if isinstance(item, Prior):
                self._priors[key] = item
            if callable(item):
                self._functions[key] = item

    def log(self, mcmc: MCMC):
        entry = {}

        for key, item in self._params.items():
            entry[key] = item[0]

        for key, item in self._priors.items():
            entry[key] = item.probability()

        for key, item in self._functions.items():
            entry[key] = item()

        entry_json = json.dumps(entry)
        self._file.write(entry_json)
        self._file.write("\n")
        self._file.flush()

    def __getstate__(self):
        return (None, self.__slots__)
