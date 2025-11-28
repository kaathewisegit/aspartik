"""
<https://beast.community/first_tutorial>
"""

from aspartik import logger
from aspartik.b3 import MCMC, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    BeastNarrowExchange,
    BeastWideExchange,
    NodeSlide,
    ParamScale,
    RootSlide,
    SubtreeSlide,
    TreeScale,
)
from aspartik.b3.parameters import Real
from aspartik.b3.priors import Bound, ConstantPopulation, Distribution
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import print_operator_stats
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import LogNormal, Uniform

logger.Logger().to_file("target/apes.trace").with_level(logger.Level.Debug).init()

msa = read_msa_from_fasta("crates/b3/data/apes.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

kappa = Real(2.0)
population_size = Real(2.0)

priors = [
    Distribution(kappa, LogNormal(1.0, 1.25)),
    Distribution(population_size, LogNormal(1.0, 1.5)),
    ConstantPopulation(tree, population_size),
]

operators = [
    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
    TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3),
    SubtreeSlide(tree, Uniform(-0.5, 0.5), rng, weight=30),
    BeastNarrowExchange(tree, rng, weight=30),
    BeastWideExchange(tree, rng, weight=3),
    RootSlide(tree, 0.75, Uniform(0, 1), rng, weight=3),
    # BEAST's `UniformOperator` picks one of the parameter dimensions moves it
    # randomly within bounds.  Using it on `nodeHeights` is equivalent to
    # selecting a random node and moving it uniformly between it's maximum and
    # minimum heights, which is what `NodeSlide` with `Uniform` does.
    NodeSlide(tree, Uniform(0, 1), rng, weight=30),
    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
]

likelihood = CPU4Likelihood(
    msa=msa,
    substitution=HKY((0.25, 0.25, 0.25, 0.25), kappa),
    clock=StrictClock(1.0),
    tree=tree,
)

loggers = [
    TreeLogger(tree=tree, path="target/apes.trees", every=1_000),
    PrintLogger(every=10_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "joint": lambda: mcmc.prior + mcmc.likelihood.likelihood(),
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.likelihood.likelihood(),
            "tree:height": lambda: tree.height_of(tree.root),
            "tree:length": lambda: tree.total_length(),
            "kappa": kappa,
            "population_size": population_size,
            "prior:kappa": priors[0],
            "prior:population_size": priors[1],
            "prior:coalescent": priors[2],
        },
        path="target/apes.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=100_000,
    state=[kappa, population_size, tree],
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)

mcmc.run()
print_operator_stats(mcmc)
