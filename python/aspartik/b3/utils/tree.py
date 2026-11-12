from ._common import raise_import

try:
    import polars as pl
except ModuleNotFoundError as e:
    raise_import(e)


from pathlib import Path
from typing import TextIO

from ...data import TaxonSet
from ...rng import RNG
from ..parameters import DatedTree


def trees_to_newick(
    names: TaxonSet,
    trace_path: str,
    dest: str | Path | TextIO,
    *,
    tree_key: str = "tree",
):
    dest = _to_io(dest)

    tree = DatedTree(names, RNG(4))
    df = pl.read_ipc(trace_path)

    for parents, heights in zip(df[f"{tree_key}:parents"], df[f"{tree_key}:heights"]):
        tree.load_parts(parents.to_numpy(), heights.to_numpy())
        dest.write(tree.to_binary().to_newick())
        dest.write("\n")

    dest.flush()


def _to_io(dest: str | Path | TextIO) -> TextIO:
    if isinstance(dest, str | Path):
        return open(dest, "w")
    else:
        return dest
