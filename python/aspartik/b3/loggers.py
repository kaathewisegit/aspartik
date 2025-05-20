"""Classes which record the state of the simulation.

All classes here adhere to the `Logger` protocol and can be passed to the `run`
function.
"""

from dataclasses import dataclass

from . import Tree


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

    def log(self, index: int):
        line = self.tree.newick()
        self._file.write(line)
        self._file.write("\n")
