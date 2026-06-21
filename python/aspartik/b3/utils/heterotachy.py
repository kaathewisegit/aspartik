from ._common import raise_import

try:
    import polars as pl
except ModuleNotFoundError as e:
    raise_import(e)


def pattern_probabilities(categories: pl.Series) -> pl.DataFrame:
    num_rows = len(categories)
    num_patterns = categories.arr.len()[0]
    n = categories.arr.max().max() + 1  # type:ignore
    name = categories.name

    return (
        categories.to_frame()
        .select(pl.col(name).arr.explode())
        .with_columns(pattern=pl.int_range(0, pl.len(), dtype=pl.UInt32) % num_patterns)
        .group_by("pattern")
        .agg([((pl.col(name) == i).sum() / num_rows).alias(f"c{i}") for i in range(n)])
        .sort("pattern")
    )


def site_probabilities(
    pattern_probs: pl.DataFrame, sites_to_patterns: list[int]
) -> pl.DataFrame:
    mapping_df = pl.select(
        site=pl.int_range(0, len(sites_to_patterns), dtype=pl.UInt32),
        pattern=pl.Series(sites_to_patterns, dtype=pl.UInt32),
    )

    return (
        mapping_df.join(pattern_probs, on="pattern", how="left")
        .drop("pattern")
        .sort("site")
    )
