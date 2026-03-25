from utils.compare import compare_verify_run

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

SCALES = [3, 30, 300]


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
    try:
        cuda_calculator = CUDALikelihood(
            msa=msa,
            substitution=HKY(frequencies, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
    except Exception:
        cuda_calculator = None

    calculators = [*cpu_calculators]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    compare_verify_run(
        "data/runs/influenza/b3.trace",
        parameters={
            "tree": tree,
            "kappa": kappa,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
        },
        likelihoods=calculators,
    )
