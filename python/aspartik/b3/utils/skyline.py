from collections.abc import Sequence
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
    num_points: int = 200,
    cred_mass: float = 0.95,
) -> None:
    _validate_num_samples(values)

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

    _plot_skyline_values(ax, boundaries, values, mode, num_points, cred_mass)


def plot_skyline_birthdeath(
    fig: plt.Figure,
    ax: plt.Axes,
    times: pl.Series | Sequence[float],
    origin: pl.Series | float,
    *,
    birth_rates: pl.Series | None = None,
    death_rates: pl.Series | None = None,
    values: pl.Series | None = None,
    times_start_from_origin: bool = True,
    mode: Mode = "traces",
    num_points: int = 200,
    cred_mass: float = 0.95,
) -> None:
    if values is None:
        if birth_rates is None or death_rates is None:
            raise ValueError("Expected values or birth_rates and death_rates")
        values = _series_ratio(birth_rates, death_rates)

    _validate_num_samples(values)

    boundaries = [
        _birthdeath_boundaries(t, o, times_start_from_origin)
        for t, o in zip(
            _iter_series_or_value(times, len(values)),
            _iter_series_or_value(origin, len(values)),
        )
    ]
    plot_values = [
        list(reversed(row)) if times_start_from_origin else list(row) for row in values
    ]

    _plot_skyline_values(
        ax,
        boundaries,
        pl.Series("values", plot_values),
        mode,
        num_points,
        cred_mass,
    )


def _validate_num_samples(values: pl.Series) -> None:
    if len(values) <= 9:
        raise ValueError("The input trace must have at least 10 samples")


def _series_ratio(numerator: pl.Series, denominator: pl.Series) -> pl.Series:
    return pl.Series(
        "values",
        [
            [n / d for n, d in zip(numerator_row, denominator_row)]
            for numerator_row, denominator_row in zip(numerator, denominator)
        ],
    )


def _iter_series_or_value(value: pl.Series | Sequence[float] | float, length: int):
    if isinstance(value, pl.Series):
        return value
    return [value] * length


def _birthdeath_boundaries(
    times: Sequence[float], origin: float, times_start_from_origin: bool
) -> list[float]:
    if times_start_from_origin:
        interval_ends = sorted(time for time in times if time > 0.0)
        interval_ends = [time for time in interval_ends if time < origin]
        interval_ends.append(origin)
        return [0.0, *[origin - time for time in reversed(interval_ends[:-1])], origin]

    interval_ends = sorted(time for time in times if 0.0 < time < origin)
    return [0.0, *interval_ends, origin]


def _plot_skyline_values(
    ax: plt.Axes,
    boundaries: list[list[float]],
    values: pl.Series,
    mode: Mode,
    num_points: int,
    cred_mass: float,
) -> None:
    if mode == "traces":
        for x, y in zip(boundaries, values):
            ax.step(x, [*y, y[-1]], where="post", color="steelblue", alpha=0.1)
    else:
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

        sorted_evals = np.sort(evals, axis=0)
        ci_range = int(np.round(cred_mass * len(sorted_evals)))
        ci_range = min(max(ci_range, 1), len(sorted_evals) - 1)

        idx = np.argmin(sorted_evals[ci_range:] - sorted_evals[:-ci_range], axis=0)
        cols = np.arange(num_points)

        ax.plot(grid, np.median(evals, axis=0), color="steelblue", label="median")
        ax.fill_between(
            grid,
            sorted_evals[idx, cols],
            sorted_evals[idx + ci_range, cols],
            color="steelblue",
            alpha=0.2,
            label=f"{cred_mass:.0%} HPD",
        )
