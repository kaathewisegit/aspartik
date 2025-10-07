"""
Based on BEAST's "Phylodynamic inference of respiratory viruses" workshop
tutorial:

<https://beast.community/workshop_respiratory_virus_phylodynamics>
"""

from datetime import datetime

from aspartik.b3 import MCMC, Likelihood, Real, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import NodeSlide, ParamScale, RandomWalk, SubtreePruneRegraft
from aspartik.b3.priors import CTMCS, Bound, Distribution, ExponentialGrowth, Yule
from aspartik.b3.substitutions import HKY
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Uniform

msa = read_msa_from_fasta("crates/b3/data/b.1.1.7.fasta").deduplicate()

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

dates = []  # tip dates
for name in msa.sequence_names():
    node = tree.leaf_by_name(name)
    assert node is not None

    date = name.split("|")[-1]
    date = datetime.strptime(date, "%Y-%m-%d").date()
    dates.append(date)

most_recent = max(dates)
for i, leaf in enumerate(tree.leaves()):
    diff = most_recent - dates[i]
    tree.set_height(leaf, diff.days / 365)

tree.set_random_heights(rng)

population_size = Real(1.0)
growth_rate = Real(0.0)
clock_rate = Real(0.001)
kappa = Real(2.0)
params = [
    population_size,
    growth_rate,
    clock_rate,
    kappa,
]

priors = [
    Bound(clock_rate, upper=1.0),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    Distribution(growth_rate, Laplace(0.0, 100.0)),
    Distribution(kappa, LogNormal(1.0, 1.25)),
    CTMCS(tree, clock_rate),
    ExponentialGrowth(tree, population_size, growth_rate),
]

operators = [
    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
    ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
    # up/down clock rate vs internal node heights
    NodeSlide(tree, Uniform(0, 1), rng, weight=1000),  # subtree leap instead
    SubtreePruneRegraft(tree, rng, weight=100),
    RandomWalk(growth_rate, window=1.0, rng=rng, weight=3),
]


sub_model = HKY((0.25, 0.25, 0.25, 0.25), kappa)
clock_model = StrictClock(1.0)
likelihood = Likelihood(
    msa=msa,
    substitution=sub_model,
    clock=clock_model,
    tree=tree,
    calculator="thread",
)

loggers = [
    TreeLogger(tree=tree, path="target/b3.trees", every=1_000),
    PrintLogger(every=1_000),
    ValueLogger(
        {
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "kappa": kappa,
            "prior.population_size": priors[-5],
            "prior.growth_rate": priors[-4],
            "prior.kappa": priors[-3],
            "prior.clock_rate": priors[-2],
            "prior.coalescent": priors[-1],
            "tree.height": lambda: tree.height_of(tree.root()),
        },
        path="target/b3.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=100_000,
    state=params + [tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    loggers=loggers,
    rng=rng,
)

mcmc.run()
