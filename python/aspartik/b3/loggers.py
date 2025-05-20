"""Classes which record the state of the simulation.

All classes here adhere to the `Logger` protocol and can be passed to the `run`
function.
"""

from dataclasses import dataclass

from . import Tree, MCMC


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
