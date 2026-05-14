from utils.compare import compare_b3

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    Likelihood,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import GTR
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

SCALES = [3, 30, 300]


def test_compare_likelihood():
    rng = RNG(4)

    msa = read_msa_from_fasta("data/alignments/H1N1pdm_2009.fasta")

    tree = Tree(msa.sequence_names(), rng)

    clock_rate = Real(1.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
    rates = RealVector(1, 1, 1, 1, 1, 1)

    cpu_calculators = [
        CPU4Likelihood(
            msa=msa,
            substitution=GTR(frequencies, rates),
            clock=Clock.Strict(clock_rate),
            tree=tree,
            scale_ln=scale,
        )
        for scale in SCALES
    ]
    try:
        cuda_calculator = CUDALikelihood(
            msa=msa,
            substitution=GTR(frequencies, rates),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
    except Exception:
        cuda_calculator = None

    calculators: list[Likelihood] = [*cpu_calculators]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    compare_b3(
        "data/runs/influenza/b3.trace",
        parameters={
            "tree": tree,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "rates": rates,
        },
        likelihoods=calculators,
    )
