"BEAST X replication tests"

import pytest
from utils.replicate import replicate_beast1

from aspartik.b3 import Calculator, Clock
from aspartik.b3.likelihoods import DNALikelihood, GammaLikelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import ConstantPopulation, ExponentialGrowth
from aspartik.b3.substitutions import GTR, HKY, JC
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def test_constant_population(rng: RNG):
    msa = read_msa_from_fasta("data/runs/test-constant/alignment.fasta")

    tree = Tree(msa.sequence_names(), rng)
    population_size = Real(1.0)

    replicate_beast1(
        "data/runs/test-constant/beast1.trace",
        "data/runs/test-constant/beast1.trees",
        parameters={
            "tree": tree,
            "population_size": population_size,
        },
        priors={"prior:coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[
            DNALikelihood(
                msa=msa,
                substitution=JC(),
                clock=Clock.Strict(Real(1.0)),
                tree=tree,
                calculator=Calculator.CPU(),
            )
        ],
    )


def test_exponential_growth(rng: RNG):
    msa = read_msa_from_fasta("data/runs/test-exponential/alignment.fasta")

    tree = Tree(msa.sequence_names(), rng)
    population_size = Real(1.0)
    growth_rate = Real(1.0)

    replicate_beast1(
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
            DNALikelihood(
                msa=msa,
                substitution=JC(),
                clock=Clock.Strict(Real(1.0)),
                tree=tree,
                calculator=Calculator.CPU(),
            )
        ],
    )


def test_hky_small(rng):
    clock_rate = Real(1.0)
    kappa = Real(2.0)
    frequencies = RealVector.repeat(0.25, 4)
    population_size = Real(1.0)

    msa = read_msa_from_fasta("data/alignments/apes.fasta")
    tree = Tree(msa.sequence_names(), rng)
    likelihood = DNALikelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CPU(),
    )

    replicate_beast1(
        "data/runs/test-hky/beast1.trace",
        "data/runs/test-hky/beast1.trees",
        parameters={
            "tree": tree,
            "clock.rate": clock_rate,
            "kappa": kappa,
            "frequencies": frequencies,
            "constant.popSize": population_size,
        },
        priors={"coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[likelihood],
    )


def test_gtr_small(rng):
    clock_rate = Real(1.0)
    rates = RealVector.repeat(0.25, 6)
    frequencies = RealVector.repeat(0.25, 4)
    population_size = Real(1.0)

    msa = read_msa_from_fasta("data/alignments/apes.fasta")
    tree = Tree(msa.sequence_names(), rng)
    likelihood = DNALikelihood(
        msa=msa,
        substitution=GTR(frequencies, rates),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CPU(),
    )

    replicate_beast1(
        "data/runs/test-gtr/beast1.trace",
        "data/runs/test-gtr/beast1.trees",
        parameters={
            "tree": tree,
            "clock.rate": clock_rate,
            "gtr.rates": rates,
            "frequencies": frequencies,
            "constant.popSize": population_size,
        },
        priors={"coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[likelihood],
    )


@pytest.mark.skip()
def test_gamma_low(rng):
    alpha = Real(0.5)
    clock_rate = Real(1.0)
    rates = RealVector.repeat(0.25, 6)
    frequencies = RealVector.repeat(0.25, 4)
    population_size = Real(1.0)

    msa = read_msa_from_fasta("data/alignments/apes.fasta")
    tree = Tree(msa.sequence_names(), rng)
    likelihood = GammaLikelihood(
        msa=msa,
        substitution=GTR(frequencies, rates),
        num_categories=4,
        alpha=alpha,
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CPU(),
    )

    replicate_beast1(
        "data/runs/test-gamma-low/beast1.trace",
        "data/runs/test-gamma-low/beast1.trees",
        parameters={
            "tree": tree,
            "alpha": alpha,
            "clock.rate": clock_rate,
            "gtr.rates": rates,
            "frequencies": frequencies,
            "constant.popSize": population_size,
        },
        priors={"coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[likelihood],
    )


def test_gamma_high(rng):
    alpha = Real(0.5)
    clock_rate = Real(1.0)
    rates = RealVector.repeat(0.25, 6)
    frequencies = RealVector.repeat(0.25, 4)
    population_size = Real(1.0)

    msa = read_msa_from_fasta("data/alignments/electricFish.fasta")
    tree = Tree(msa.sequence_names(), rng)
    likelihood = GammaLikelihood(
        msa=msa,
        substitution=GTR(frequencies, rates),
        num_categories=4,
        alpha=alpha,
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CPU(),
    )

    replicate_beast1(
        "data/runs/test-gamma-high/beast1.trace",
        "data/runs/test-gamma-high/beast1.trees",
        parameters={
            "tree": tree,
            "alpha": alpha,
            "clock.rate": clock_rate,
            "gtr.rates": rates,
            "frequencies": frequencies,
            "constant.popSize": population_size,
        },
        priors={"coalescent": ConstantPopulation(tree, population_size)},
        likelihoods=[likelihood],
    )
