from .. import State


class TreeLogger:
    def __init__(self, path, every: int):
        self.file = open(path, "w")
        self.every = every

    def log(self, state: State, _index: int):
        line = state.tree.newick()
        self.file.write(line)
        self.file.write("\n")
