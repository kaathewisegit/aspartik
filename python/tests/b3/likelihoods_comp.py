from utils.compare import compare

from aspartik.b3 import MCMC, Clock, Prior
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    HeteroLikelihood,
    Likelihood,
    Parallel4Likelihood,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth
from aspartik.b3.substitutions import HKY
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Uniform


def test_compare_likelihood():
    rng = RNG(4)

    msa = read_msa_from_fasta("data/alignments/influenza.fasta")

    tree = Tree(msa.sequence_names(), rng)

    kappa = Real(0)
    population_size = Real(0)
    growth_rate = Real(0)
    clock_rate = Real(0)
    frequencies = RealVector(0, 0, 0, 0)

    cpu_calculator = CPU4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
    )
    parallel_calculator = Parallel4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        num_leaf_threads=5,
        num_internal_threads=2,
    )
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
    except Exception as e:
        cuda_calculator = None

    calculators = [cpu_calculator, parallel_calculator, hetero]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    compare(
        "data/runs/influenza/b3.trace",
        "data/runs/influenza/b3.trees",
        parameters={
            "kappa": kappa,
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
        },
        likelihoods=calculators,
    )
