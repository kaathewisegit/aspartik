from .. import State, Tree


class TreeLogger:
    def __init__(self, tree: Tree, path: str, every: int):
        self.tree = tree
        self.file = open(path, "w")
        self.every = every

    def log(self, state: State, index: int):
        line = self.tree.newick()
        self.file.write(line)
        self.file.write("\n")
