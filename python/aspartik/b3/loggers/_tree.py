from .. import Tree


class TreeLogger:
    """Records the topology of the tree into a `.trees` file."""

    every: int
    """How often the logger will be called"""

    def __init__(self, tree: Tree, path: str, every: int):
        """
        Args:
            tree:
            path:
                Path to the file where the trees will be appended in Newick
                format, one per line.  It's opened verbatim (the `.trees`
                extension won't be added).
            every:
                The logger will record a new tree after `every` steps.
        """
        self._tree = tree
        self._file = open(path, "w")
        self.every = every

    def log(self, index: int):
        line = self._tree.newick()
        self._file.write(line)
        self._file.write("\n")
