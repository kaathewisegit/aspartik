"""Classes which record the state of the simulation.

All classes here adhere to the `Logger` protocol and can be passed to the `run`
function.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import json

from . import MCMC, Tree, Parameter, Prior


@dataclass
class TreeLogger:
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

    def log(self, mcmc: MCMC, index: int):
        line = self.tree.newick()
        self._file.write(line)
        self._file.write("\n")


@dataclass
class PrintLogger:
    every: int

    def __post_init__(self):
        print(f"{'step':>16}{'posterior':>16}{'likelihood':>16}{'prior':>16}")

    def log(self, mcmc: MCMC, index: int):
        print(
            f"{index:>16}{mcmc.posterior:>16.2f}{mcmc.likelihood:>16.2f}{mcmc.prior:>16.2f}"
        )


@dataclass
class ValueLogger:
    map: Mapping[str, Parameter | Prior]
    path: str
    every: int

    def __post_init__(self):
        self._file = open(self.path, "w")
        self._params = {}
        self._priors = {}

        for key, item in self.map.items():
            if isinstance(item, Parameter):
                self._params[key] = item
            if isinstance(item, Prior):
                self._priors[key] = item

    def log(self, mcmc: MCMC, index: int):
        entry = {}

        for key, item in self._params.items():
            entry[key] = item[0]

        for key, item in self._priors.items():
            entry[key] = item.probability()

        entry_json = json.dumps(entry)
        self._file.write(entry_json)
        self._file.write("\n")
