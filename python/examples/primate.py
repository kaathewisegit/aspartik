"""
<https://beast.community/first_tutorial>
"""

from aspartik.b3 import MCMC, Likelihood, Real, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    NarrowExchange,
    NodeSlide,
    ParamScale,
    RootSlide,
    SubtreeSlide,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.b3.priors import Bound, Distribution, Yule
from aspartik.b3.substitutions import HKY, JC
from aspartik.b3.utils import print_operator_stats
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("crates/b3/data/primate.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

kappa = Real(2.0)
yule_birth_rate = Real(1.0)

priors = [
    Distribution(kappa, LogNormal(1.0, 1.25)),
    Distribution(yule_birth_rate, LogNormal(1.0, 1.25)),
    Yule(tree, yule_birth_rate),
]

operators = [
    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
    TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3),
    SubtreeSlide(tree, Normal(0, 1), rng, weight=30),
    NarrowExchange(tree, rng, weight=30),
    WideExchange(tree, rng, weight=3),
    WilsonBalding(tree, rng, weight=3),
    RootSlide(tree, 0.75, Uniform(0, 1), rng, weight=3),
    # BEAST's `UniformOperator` picks one of the parameter dimensions moves it
    # randomly within bounds.  Using it on `nodeHeights` is equivalent to
    # selecting a random node and moving it uniformly between it's maximum and
    # minimum heights, which is what `NodeSlide` with `Uniform` does.
    NodeSlide(tree, Uniform(0, 1), rng, weight=30),
    ParamScale(yule_birth_rate, 0.75, Uniform(0, 1), rng, weight=3),
]

likelihood = Likelihood(
    msa=msa,
    substitution=HKY((0.25, 0.25, 0.25, 0.25), kappa),
    clock=StrictClock(1.0),
    tree=tree,
    calculator="cpu",
)

loggers = [
    TreeLogger(tree=tree, path="target/primate.trees", every=1_000),
    PrintLogger(every=10_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "joint": lambda: mcmc.prior + mcmc.likelihood,
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.likelihood,
            "tree:root_height": lambda: tree.height_of(tree.root),
            "kappa": kappa,
            "yule_birth_rate": yule_birth_rate,
        },
        path="target/primate.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=1_000_000,
    state=[kappa, yule_birth_rate, tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    callbacks=loggers,
    rng=rng,
)

mcmc.run()
print_operator_stats(mcmc)
