from typing import Literal

from ._common import raise_import

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
except ModuleNotFoundError as e:
    raise_import(e)

from aspartik.b3.parameters import Tree
from aspartik.rng import RNG

Mode = Literal["traces", "hpd"]


def plot_skyline_coalescent(
    fig: plt.Figure,
    ax: plt.Axes,
    trees: pl.Series,
    group_sizes: pl.Series,
    values: pl.Series,
    sequence_names: list[str],
    mode: Mode = "traces",
    *,
    num_points: int = 200,
    cred_mass: float = 0.95,
) -> None:
    _validate_num_samples(values, cred_mass)

    rng = RNG(4)
    tree = Tree(sequence_names, rng)

    num_groups = group_sizes.arr.len()[0]
    num_heights = len(tree.internal_heights())

    boundaries = []
    for tree_state, group_size in zip(trees, group_sizes):
        tree.load(tree_state)
        internal_heights = tree.internal_heights()

        n = 0
        group_times = []
        for i in range(num_groups):
            n += group_size[i]
            group_times.append(internal_heights[min(n, num_heights - 1)])
        boundaries.append([0.0, *group_times])

    if mode == "traces":
        for x, y in zip(boundaries, values):
            ax.step(x, [*y, y[-1]], where="post", color="steelblue", alpha=0.1)
        return

    max_height = max(boundary[-1] for boundary in boundaries)
    grid = np.linspace(0.0, max_height, num_points)

    evals = np.array(
        [
            np.asarray(y)[
                np.searchsorted(x[1:], grid, side="right").clip(max=len(y) - 1)
            ]
            for x, y in zip(boundaries, values)
        ]
    )

    low, high = _hpd_2d(evals, cred_mass)
    ax.plot(grid, np.median(evals, axis=0), color="steelblue", label="median")
    ax.fill_between(
        grid, low, high, color="steelblue", alpha=0.2, label=f"{cred_mass:.0%} HPD"
    )


def plot_skyline_birthdeath(
    fig: plt.Figure,
    ax: plt.Axes,
    reproductive_number: pl.Series,
    *,
    origin: pl.Series | None = None,
    interval_times: pl.Series | None = None,
    num_points: int = 200,
    cred_mass: float = 0.95,
) -> None:
    _validate_num_samples(reproductive_number, cred_mass)

    if origin is None and interval_times is None:
        raise ValueError(
            "at least one of `origin` or `interval_times` must be provided"
        )

    if origin is not None:
        max_time = origin.cast(pl.Float64).max()
    elif interval_times is not None:
        n = len(reproductive_number[0])
        max_time = (interval_times.cast(pl.Float64) * n).max()
    assert isinstance(max_time, float)

    grid = np.linspace(0.0, max_time, num_points)
    evals = np.empty((len(reproductive_number), num_points))

    for i, row in enumerate(reproductive_number):
        n = len(row)
        if origin is not None:
            mt = float(origin[i])
        elif interval_times is not None:
            mt = float(interval_times[i]) * n
        else:
            raise AssertionError
        if interval_times is not None:
            width = float(interval_times[i])
        else:
            width = mt / n
        vals = np.asarray(row, dtype=float)

        boundaries = np.empty(n + 1)
        boundaries[:n] = width * np.arange(n)
        boundaries[n] = mt

        idx = np.searchsorted(boundaries, mt - grid, side="right") - 1
        np.clip(idx, 0, n - 1, out=idx)
        evals[i] = vals[idx]

    low, high = _hpd_2d(evals, cred_mass)
    ax.plot(grid, np.median(evals, axis=0), color="darkorange", label=r"median $R_e$")
    ax.fill_between(
        grid,
        low,
        high,
        color="darkorange",
        alpha=0.2,
        label=rf"$R_e$ {cred_mass:.0%} HPD",
    )

    ax.set_ylabel(r"$R_e$")
    ax.tick_params(axis="y")
    ax.spines["left"]

    ax.set_xlim(max_time, 0.0)

    ax.legend()


def _validate_num_samples(values: pl.Series | np.ndarray, cred_mass: float) -> None:
    if not 0.0 < cred_mass < 1.0:
        raise ValueError("cred_mass must be strictly between 0 and 1")
    min_samples = int(np.ceil(1.0 / (1.0 - cred_mass)))
    if len(values) < min_samples:
        raise ValueError(
            f"The input trace must have at least {min_samples} samples "
            f"for a {cred_mass:.0%} credible interval"
        )


def _hpd_1d(samples: np.ndarray, cred_mass: float) -> tuple[float, float]:
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    ci_range = int(np.round(cred_mass * n))
    ci_range = min(max(ci_range, 1), n - 1)
    widths = sorted_samples[ci_range:] - sorted_samples[:-ci_range]
    idx = int(np.argmin(widths))
    return float(sorted_samples[idx]), float(sorted_samples[idx + ci_range])


def _hpd_2d(evals: np.ndarray, cred_mass: float) -> tuple[np.ndarray, np.ndarray]:
    sorted_evals = np.sort(evals, axis=0)
    n = len(sorted_evals)
    ci_range = int(np.round(cred_mass * n))
    ci_range = min(max(ci_range, 1), n - 1)
    idx = np.argmin(sorted_evals[ci_range:] - sorted_evals[:-ci_range], axis=0)
    cols = np.arange(evals.shape[1])
    return sorted_evals[idx, cols], sorted_evals[idx + ci_range, cols]
