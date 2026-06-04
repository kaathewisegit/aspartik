import polars as pl


def siteprobs(categories: pl.Series) -> pl.DataFrame:
    num_rows = len(categories)
    num_sites = categories.arr.len()[0]
    n = categories.arr.max().max() + 1  # type:ignore
    name = categories.name

    return (
        categories.to_frame()
        .select(pl.col(name).arr.explode())
        .with_columns(site=pl.int_range(0, pl.len(), dtype=pl.UInt32) % num_sites)
        .group_by("site")
        .agg([((pl.col(name) == i).sum() / num_rows).alias(f"c{i}") for i in range(n)])
        .sort("site")
    )
