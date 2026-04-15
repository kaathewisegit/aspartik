import polars as pl

from typing import Literal

from ...io import read_msa_from_fasta
from ..config import MCMCConfig


def burnin(df, share: float = 0.5):
    return df.iloc[int(len(df) * share) :]


def assert_value_close(b3, beast, col: str, threshold: float = 0.05):
    b3_mean = b3[col].mean()
    beast_mean = beast[col].mean()
    diff = abs(b3_mean - beast_mean) / beast_mean

    assert diff < threshold, f"{col}: {b3_mean} vs {beast_mean}"


def compare_beast1(
    fasta_path: str,
    length: int,
    model: Literal["HKY"],
    tree_prior: Literal["yule", "constant"],
):
    name = f"{tree_prior}_{model}"
    b3_path = f"target/{name}.trace"
    beast1_path = f"target/{name}.beast1.log"

    b3_c = MCMCConfig(
        msa=read_msa_from_fasta(fasta_path),
        substitution_model="HKY",
        tree_prior=tree_prior,
        length=length,
        calculator="cpu",
        trace_path=b3_path,
    )
    b3_c.b3_make_and_run()

    beast1_c = MCMCConfig(
        msa=read_msa_from_fasta(fasta_path),
        substitution_model="HKY",
        tree_prior=tree_prior,
        length=length,
        calculator="cpu",
        trace_path=beast1_path,
    )
    beast1_c.beast1_make_and_run()

    b3 = burnin(pl.read_ipc(b3_path))
    beast1 = burnin(pl.read_csv(beast1_path, separator="\t", skip_lines=3))

    columns = ["clock_rate"]
    match model:
        case "HKY":
            columns.append("kappa")

    match tree_prior:
        case "yule":
            columns.append("birth_rate")
        case "constant":
            columns.append("population_size")

    for column_name in columns:
        assert_value_close(b3, beast1, column_name)
