"""
<https://beast.community/first_tutorial>
"""

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    BeastNarrowExchange,
    BeastWideExchange,
    DeltaExchange,
    NodeSlide,
    ParamScale,
    RootSlide,
    SubtreeSlide,
    TreeScale,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import ConstantPopulation, Distribution
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, LogNormal, Uniform


def make_mcmc(fasta_path: str):
    msa = read_msa_from_fasta(fasta_path)

    rng = RNG(4)
    tree = Tree(msa.sequence_names(), rng)

    kappa = Real(2.0)
    population_size = Real(2.0)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)

    priors = [
        Distribution(kappa, LogNormal(1.0, 1.25)),
        Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
        ConstantPopulation(tree, population_size),
    ]

    operators = [
        ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
        DeltaExchange(frequencies, factor=0.01, rng=rng, weight=1),
        TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3),
        SubtreeSlide(tree, Uniform(-0.5, 0.5), rng, weight=30),
        BeastNarrowExchange(tree, rng, weight=30),
        BeastWideExchange(tree, rng, weight=3),
        RootSlide(tree, 0.75, Uniform(0, 1), rng, weight=3),
        # BEAST's `UniformOperator` picks one of the parameter dimensions moves it
        # randomly within bounds.  Using it on `nodeHeights` is equivalent to
        # selecting a random node and moving it uniformly between it's maximum and
        # minimum heights, which is what `NodeSlide` with `Uniform` does.
        NodeSlide(tree, rng, weight=30),
        ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
    ]

    likelihood = CPU4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(Real(1.0)),
        tree=tree,
    )

    loggers = [
        TreeLogger(tree=tree, path="target/apes.trees", every=1_000),
        PrintLogger(every=10_000),
        ValueLogger(
            {
                "step": lambda: mcmc.current_step,
                "posterior": lambda: mcmc.posterior,
                "prior": lambda: mcmc.prior,
                "likelihood": lambda: mcmc.likelihood.likelihood(),
                "tree:height": lambda: tree.height_of(tree.root),
                "tree:length": lambda: tree.total_length(),
                "kappa": kappa,
                "population_size": population_size,
                "frequencies": frequencies,
                "prior:kappa": priors[0],
                "prior:population_size": priors[1],
                "prior:coalescent": priors[2],
            },
            path="target/apes.log",
            every=1_000,
        ),
        TraceWriter(
            {
                "kappa": kappa,
                "population_size": population_size,
                "frequencies": frequencies,
                "tree": tree,
                "clock_rate": Real(1.0),
            },
            "target/apes.trace",
            overwrite=True,
            zstd=True,
            every=1_000,
        ),
    ]

    mcmc = MCMC(
        state=[tree, kappa, population_size, frequencies],
        priors=priors,
        operators=operators,
        likelihood=likelihood,
        callbacks=loggers,
        rng=rng,
    )

    return mcmc


def run(fasta_path: str):
    run_from_cmdline(make_mcmc(fasta_path))


if __name__ == "__main__":
    run("data/alignments/apes.fasta")
