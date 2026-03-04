import pandas as pd

from typing import Literal

from ...io import read_msa_from_fasta
from .beast import beast1_config, beast1_run
from .config import make_mcmc


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

    msa = read_msa_from_fasta(fasta_path)
    b3 = make_mcmc(
        msa,
        trace_path=b3_path,
        substitution_model="HKY",
        tree_prior=tree_prior,
    )
    b3.run(length)

    beast1_xml_config = beast1_config(
        msa,
        log_path=beast1_path,
        substitution_model="HKY",
        tree_prior=tree_prior,
        length=length,
    )
    beast1_run(beast1_xml_config, "cpu")

    b3 = burnin(pd.read_feather(b3_path))
    beast1 = burnin(pd.read_csv(beast1_path, sep="\t", skiprows=3))

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
