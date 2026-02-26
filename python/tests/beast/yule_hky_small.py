"""
<https://beast.community/first_tutorial>
"""

import pandas as pd
import pytest

from aspartik.b3.utils import print_operator_stats
from aspartik.b3.utils.beast import beast1_config, beast1_run
from aspartik.b3.utils.config import make_mcmc
from aspartik.io.msa import read_msa_from_fasta


def burnin(df, share: float = 0.5):
    return df.iloc[int(len(df) * share) :]


@pytest.mark.manual
def test():
    compare("data/alignments/apes.fasta", 10_000_000)


def compare(fasta_path: str, length: int):
    b3_path = "target/yule_hky_small.trace"
    beast1_path = "target/yule_hky_small.beast1.log"

    msa = read_msa_from_fasta(fasta_path)
    b3 = make_mcmc(
        msa,
        trace_path=b3_path,
        substitution_model="HKY",
        tree_prior="yule",
    )
    b3.run(length)
    print_operator_stats(b3)

    beast1_xml_config = beast1_config(
        msa,
        log_path=beast1_path,
        substitution_model="HKY",
        tree_prior="yule",
        length=length,
    )
    beast1_run(beast1_xml_config, "cpu")

    b3 = burnin(pd.read_feather(b3_path))
    beast1 = burnin(pd.read_csv(beast1_path, sep="\t", skiprows=3))

    print(b3)
    print(beast1)

    print(b3["clock_rate"].mean())
    print(beast1["clock_rate"].mean())
