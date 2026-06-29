"""
<https://beast.community/first_tutorial>
"""

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import (
    OperatorStats,
    PrintLogger,
    StateCheckpoint,
    TraceWriter,
)
from aspartik.b3.likelihoods import DNALikelihood
from aspartik.b3.operators import (
    BeastNarrowExchange,
    BeastWideExchange,
    DeltaExchange,
    NodeSlide,
    RootSlide,
    ScaleReal,
    SubtreeSlide,
    TreeScale,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import ConstantPopulation, Distribution
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, LogNormal, Uniform

msa = read_msa_from_fasta("data/alignments/apes.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

kappa = Real(2.0)
population_size = Real(2.0)
frequencies = RealVector.repeat(0.25, 4)

priors = [
    Distribution(kappa, LogNormal(1.0, 1.25)),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    ConstantPopulation(tree, population_size),
]

operators = [
    ScaleReal(kappa, Uniform(0, 1), rng, weight=1),
    DeltaExchange(frequencies, rng=rng, weight=1),
    TreeScale(tree, Uniform(0, 1), rng, weight=3),
    SubtreeSlide(tree, Uniform(-0.5, 0.5), rng, weight=30),
    BeastNarrowExchange(tree, rng, weight=30),
    BeastWideExchange(tree, rng, weight=3),
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    # BEAST's `UniformOperator` picks one of the parameter dimensions moves it
    # randomly within bounds.  Using it on `nodeHeights` is equivalent to
    # selecting a random node and moving it uniformly between it's maximum and
    # minimum heights, which is what `NodeSlide` with `Uniform` does.
    NodeSlide(tree, rng, weight=30),
    ScaleReal(population_size, Uniform(0, 1), rng, weight=3),
]

likelihood = DNALikelihood(
    msa=msa,
    substitution=HKY(frequencies, kappa),
    clock=Clock.Strict(Real(1.0)),
    tree=tree,
    calculator=Calculator.CPU(),
)

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "kappa": kappa,
            "population_size": population_size,
            "frequencies": frequencies,
            "tree": tree,
            "prior:kappa": priors[0],
            "prior:population_size": priors[1],
            "prior:coalescent": priors[2],
        },
        "target/apes.trace",
        overwrite=True,
        zstd=True,
        every=1_000,
    ),
    StateCheckpoint("target/apes.state", every=10_000),
    OperatorStats("target/apes.opstats", every=100_000),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
    optimization_cutoff=100_000,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
