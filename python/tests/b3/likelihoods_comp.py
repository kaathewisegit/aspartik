import pytest
from utils.compare import compare

import os

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    HeteroLikelihood,
    Parallel4Likelihood,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG

SCALES = [3, 30, 300]


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Expensive")
def test_compare_likelihood():
    rng = RNG(4)

    msa = read_msa_from_fasta("data/alignments/influenza.fasta")

    tree = Tree(msa.sequence_names(), rng)

    kappa = Real(1.0)
    clock_rate = Real(1.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)

    cpu_calculators = [
        CPU4Likelihood(
            msa=msa,
            substitution=HKY(frequencies, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
            scale_ln=scale,
        )
        for scale in SCALES
    ]
    parallel_calculators = [
        Parallel4Likelihood(
            msa=msa,
            substitution=HKY(frequencies, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
            num_leaf_threads=5,
            num_internal_threads=2,
            scale_ln=scale,
        )
        for scale in SCALES
    ]
    hetero = HeteroLikelihood(
        likelihoods=[
            CPU4Likelihood(
                msa=msa,
                substitution=HKY(frequencies, kappa),
                clock=Clock.Strict(clock_rate),
                tree=tree,
            )
        ]
    )
    try:
        cuda_calculator = CUDALikelihood(
            msa=msa,
            substitution=HKY(frequencies, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
    except Exception:
        cuda_calculator = None

    calculators = [*cpu_calculators, *parallel_calculators, hetero]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    compare(
        "data/runs/influenza/b3.trace",
        parameters={
            "tree": tree,
            "kappa": kappa,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
        },
        likelihoods=calculators,
    )
