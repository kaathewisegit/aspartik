from aspartik.b3 import MCMC, Likelihood, Real, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.loggers import PrintLogger
from aspartik.b3.operators import (
    EpochScale,
    NarrowExchange,
    NodeSlide,
    ParamScale,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.b3.priors import Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import print_operator_stats
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Uniform

msa = read_msa_from_fasta("crates/b3/data/sim.fasta").deduplicate()

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(rng)

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
    msa=msa,
    substitution=sub_model,
    clock=clock_model,
    tree=tree,
    calculator="cuda",
)

loggers = [
    PrintLogger(every=1_000),
]

mcmc = MCMC(
    burnin=0,
    length=1_000_000,
    state=[birth_rate_y, tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    loggers=loggers,
    rng=rng,
)

mcmc.run()

print_operator_stats(mcmc)
