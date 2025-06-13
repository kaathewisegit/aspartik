from aspartik.b3 import MCMC, Tree, Real, Likelihood
from aspartik.b3.loggers import PrintLogger
from aspartik.b3.operators import (
    ParamScale,
    EpochScale,
    TreeScale,
    NarrowExchange,
    WideExchange,
    NodeSlide,
    WilsonBalding,
)
from aspartik.b3.priors import Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.clocks import StrictClock
from aspartik.stats.distributions import Uniform
from aspartik.rng import RNG
from aspartik.io.fasta import DNAReader

path = "crates/b3/data/sim.fasta"
sequences = []
names = []
for i, record in enumerate(DNAReader(path)):
    if i == 500:
        break
    sequences.append(record.sequence)
    names.append(record.id)

rng = RNG(4)
tree = Tree(names, rng)

birth_rate_y = Real(2.0)

priors = [
    Yule(tree, birth_rate_y),
]

operators = [
    ParamScale(birth_rate_y, 0.1, Uniform(0, 1), rng, weight=3),
    EpochScale(tree, 0.9, Uniform(0, 1), rng, weight=4.0),
    TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=2.0),
    NodeSlide(tree, Uniform(0, 1), rng, weight=45.0),
    NarrowExchange(tree, rng, weight=15.0),
    WideExchange(tree, rng, weight=3.0),
    WilsonBalding(tree, rng, weight=3.0),
]

sub_model = HKY((0.25, 0.25, 0.25, 0.25), 2.0)
clock_model = StrictClock(1.0)
likelihood = Likelihood(
    sequences=sequences,
    substitution=sub_model,
    clock=clock_model,
    tree=tree,
    calculator="thread",
)

loggers = [
    PrintLogger(every=1_000),
]

mcmc = MCMC(
    burnin=0,
    length=100_000,
    state=[birth_rate_y, tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    loggers=loggers,
    rng=rng,
)

mcmc.run()
