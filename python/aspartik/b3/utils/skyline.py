from ._common import raise_import

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
except ModuleNotFoundError as e:
    raise_import(e)


from typing import Literal

from aspartik.b3.parameters import Tree
from aspartik.rng import RNG


def plot_skyline(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pl.DataFrame,
    pop_sizes_col: str,
    sequence_names: list[str],
    mode: Literal["traces", "hpd"] = "traces",
    num_points: int = 200,
    cred_mass: float = 0.95,
) -> None:
    if len(df) <= 9:
        raise ValueError("The input trace must have at least 10 samples")

    rng = RNG(4)
    tree = Tree(sequence_names, rng)

    num_groups = df["group_sizes"].arr.len()[0]
    num_heights = len(tree.internal_heights())

    times = []
    for row in df.iter_rows(named=True):
        tree.load(row["tree"])
        internal_heights = tree.internal_heights()

        n = 0
        group_times = []
        for i in range(num_groups):
            n += row["group_sizes"][i]
            group_times.append(internal_heights[min(n, num_heights - 1)])
        times.append(group_times)

    heights = pl.Series("heights", times, dtype=pl.Array(pl.Float64, num_groups))
    pop_sizes = df[pop_sizes_col]

    if mode == "traces":
        for h, p in zip(heights, pop_sizes):
            x = [0, *list(h)]
            y = [*p, p[-1]]
            ax.step(x, y, where="post", color="steelblue", alpha=0.1)
    else:
        max_height = heights.explode().max()
        assert max_height is float
        grid = np.linspace(0.0, max_height, num_points)

        evals = np.array(
            [
                p[np.searchsorted(h, grid, side="right").clip(max=len(p) - 1)]
                for h, p in zip(heights, pop_sizes)
            ]
        )

        sorted_evals = np.sort(evals, axis=0)
        ci_range = int(np.round(cred_mass * len(sorted_evals)))

        idx = np.argmin(sorted_evals[ci_range:] - sorted_evals[:-ci_range], axis=0)
        cols = np.arange(num_points)

        ax.fill_between(
            grid,
            sorted_evals[idx, cols],
            sorted_evals[idx + ci_range, cols],
            color="steelblue",
            alpha=0.2,
        )
        ax.plot(grid, np.median(evals, axis=0), color="steelblue")
