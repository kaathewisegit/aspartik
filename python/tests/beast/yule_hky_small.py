"""
<https://beast.community/first_tutorial>
"""

import pandas as pd
import pytest

from aspartik.b3.utils.beast import beast1_config, beast1_run
from aspartik.b3.utils.config import make_mcmc
from aspartik.io.msa import read_msa_from_fasta


def burnin(df, share: float = 0.5):
    return df.iloc[int(len(df) * share) :]


@pytest.mark.manual
def test():
    compare("data/alignments/apes.fasta", 10_000_000)


def compare(fasta_path: str, length: int):
    msa = read_msa_from_fasta(fasta_path)
    b3 = make_mcmc(
        msa,
        trace_path="target/test.trace",
        substitution_model="HKY",
        tree_prior="constant",
    )
    b3.run(length)

    beast1_xml_config = beast1_config(
        msa,
        log_path="target/test.beast1.log",
        substitution_model="HKY",
        tree_prior="constant",
        length=length,
    )
    beast1_run(beast1_xml_config, "cpu")

    b3 = burnin(pd.read_feather("target/test.trace"))
    beast1 = burnin(pd.read_csv("target/test.beast1.log", sep="\t", skiprows=3))

    print(b3)
    print(beast1)

    print(b3["clock_rate"].mean())
    print(beast1["clock_rate"].mean())

    raise Exception
