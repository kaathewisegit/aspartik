from pathlib import Path
from typing import TextIO

from ...rng import RNG
from ..parameters import Tree


def trees_to_newick(
    names: list[str],
    trace_path: str,
    dest: str | Path | TextIO,
    *,
    tree_key: str = "tree",
):
    import pandas as pd

    dest = _to_io(dest)

    tree = Tree(names, RNG(4))
    df = pd.read_feather(trace_path)

    for bytes in df["tree"]:
        tree.load(bytes)
        dest.write(tree.newick())
        dest.write("\n")

    dest.flush()


def _to_io(dest: str | Path | TextIO) -> TextIO:
    if isinstance(dest, str | Path):
        return open(dest, "w")
    else:
        return dest
