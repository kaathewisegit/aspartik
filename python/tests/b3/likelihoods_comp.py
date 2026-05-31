from utils.replicate import replicate_b3

from aspartik.b3 import Calculator, Clock
from aspartik.b3.likelihoods import DNALikelihood, Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import GTR
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

SCALES = [3, 30, 300]


def test_replicate_likelihood():
    rng = RNG(4)

    msa = read_msa_from_fasta("data/alignments/H1N1pdm_2009.fasta")

    tree = Tree(msa.sequence_names(), rng)

    clock_rate = Real(1.0)
    frequencies = RealVector.repeat(0.25, 4)
    rates = RealVector.repeat(1, 6)

    cpu_calculators = [
        DNALikelihood(
            msa=msa,
            substitution=GTR(frequencies, rates),
            clock=Clock.Strict(clock_rate),
            tree=tree,
            calculator=Calculator.CPU().with_scale(scale),
        )
        for scale in SCALES
    ]
    try:
        cuda_calculator = DNALikelihood(
            msa=msa,
            substitution=GTR(frequencies, rates),
            clock=Clock.Strict(clock_rate),
            tree=tree,
            calculator=Calculator.CUDA(),
        )
    except Exception:
        cuda_calculator = None

    calculators: list[Likelihood] = [*cpu_calculators]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    replicate_b3(
        "data/runs/influenza/b3.trace",
        parameters={
            "tree": tree,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "rates": rates,
        },
        likelihoods=calculators,
    )
