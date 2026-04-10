from utils.compare import compare_beast1

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.parameters import Real, Tree
from aspartik.b3.priors import ConstantPopulation, ExponentialGrowth
from aspartik.b3.substitutions import JC
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def test_constant_population(rng: RNG):
    msa = read_msa_from_fasta("data/runs/test-constant/alignment.fasta")

    tree = Tree(msa.sequence_names(), rng)
    population_size = Real(1.0)

    compare_beast1(
        "data/runs/test-constant/beast1.trace",
        "data/runs/test-constant/beast1.trees",
        parameters={
            "tree": tree,
            "population_size": population_size,
        },
        priors={"prior:coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[
            CPU4Likelihood(
                msa=msa,
                substitution=JC(),
                clock=Clock.Strict(Real(1.0)),
                tree=tree,
            )
        ],
    )


def test_exponential_growth(rng: RNG):
    msa = read_msa_from_fasta("data/runs/test-exponential/alignment.fasta")

    tree = Tree(msa.sequence_names(), rng)
    population_size = Real(1.0)
    growth_rate = Real(1.0)

    compare_beast1(
        "data/runs/test-exponential/beast1.trace",
        "data/runs/test-exponential/beast1.trees",
        parameters={
            "tree": tree,
            "population_size": population_size,
            "growth_rate": growth_rate,
        },
        priors={
            "prior:coalescent": ExponentialGrowth(tree, population_size, growth_rate)
        },
        likelihoods=[
            CPU4Likelihood(
                msa=msa,
                substitution=JC(),
                clock=Clock.Strict(Real(1.0)),
                tree=tree,
            )
        ],
    )
