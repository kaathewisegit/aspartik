"""
Based on BEAST's "Phylodynamic inference of respiratory viruses" workshop
tutorial:

<https://beast.community/workshop_respiratory_virus_phylodynamics>
"""

from collections import deque
from datetime import datetime
from pathlib import Path

from aspartik.b3 import MCMC, Likelihood, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    BeastNarrowExchange,
    NodeSlide,
    ParamScale,
    RandomWalk,
    RootSlide,
    SubtreeLeap,
    SubtreePruneRegraft,
    SubtreeSlide,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real
from aspartik.b3.priors import Distribution, ExponentialGrowth
from aspartik.b3.substitutions import K80
from aspartik.b3.tree import Internal
from aspartik.b3.utils import print_operator_stats
from aspartik.data import newick
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("crates/b3/data/b.1.1.7.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

starting = newick.Tree(Path("crates/b3/data/respiratory-starting.tree").read_text())
tree = Tree.from_newick(starting)
tree.validate()


kappa = Real(2.0)
population_size = Real(1.0)
growth_rate = Real(0.0)
clock_rate = Real(0.001)
params = [kappa, population_size, growth_rate, clock_rate]

priors = [
    Distribution(kappa, LogNormal(1.0, 1.25)),
    Distribution(clock_rate, Laplace(0, 0.5)),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    Distribution(growth_rate, Laplace(0.0, 100.0)),
    ExponentialGrowth(tree, population_size, growth_rate),
]

operators = [
    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
    ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
    UpDown(Internals(tree), clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=30),
    SubtreePruneRegraft(tree, rng, weight=30),
    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
    RandomWalk(growth_rate, window=1.0, rng=rng, weight=3),
    #
    BeastNarrowExchange(tree, rng, weight=30),
    RootSlide(tree, 0.75, Uniform(0, 1), rng, weight=3),
    NodeSlide(tree, Uniform(0, 1), rng, weight=30),
]


likelihood = Likelihood(
    msa=msa,
    substitution=K80(kappa),
    clock=StrictClock(clock_rate),
    tree=tree,
    calculator="cuda",
    cuda_device=1,
)

loggers = [
    TreeLogger(tree=tree, path="target/b3.trees", every=1_000),
    PrintLogger(every=1_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "joint": lambda: mcmc.prior + mcmc.cached_likelihood,
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.cached_likelihood,
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "kappa": kappa,
            "prior:kappa": priors[0],
            "prior:clock_rate": priors[1],
            "prior:population_size": priors[2],
            "prior:growth_rate": priors[3],
            "prior:coalescent": priors[4],
            "tree:height": lambda: tree.height_of(tree.root),
            "tree:length": lambda: tree.total_length(),
        },
        path="target/b3.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=10_000_000,
    state=params + [tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    callbacks=loggers,
    rng=rng,
)


try:
    mcmc.run()
except Exception as e:
    print(e)

print_operator_stats(mcmc)
