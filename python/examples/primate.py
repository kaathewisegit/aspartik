from aspartik.b3 import MCMC, Likelihood, Real, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    DeltaExchange,
    EpochScale,
    NarrowExchange,
    NodeSlide,
    ParamScale,
    SubtreePruneRegraft,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.b3.priors import ConstantPopulation, Distribution, Monophyly, Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import print_operator_stats
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Exp, Gamma, LogNormal, Uniform

msa = read_msa_from_fasta("crates/b3/data/primate-mdna-full.fasta").deduplicate()

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(rng)

mutation_rate = Real(1.0)
gamma_shape = Real(1.0)
kappa = Real(2.0)

population = Real(1.0)
birth_rate_y = Real(1.0)

params = [
    mutation_rate,
    gamma_shape,
    kappa,
    birth_rate_y,
]


homo, pan = tree.leaf_by_name("Homo_sapiens"), tree.leaf_by_name("Pan_troglodytes")
assert homo != None
assert pan != None

priors = [
    Yule(tree, birth_rate_y),
    Distribution(birth_rate_y, Gamma(0.001, 1 / 1000.0)),
    Distribution(gamma_shape, Exp(1.0)),
    Distribution(kappa, LogNormal(1.0, 1.25)),
    ConstantPopulation(tree, population),
    Monophyly(tree, [homo, pan]),
]

# TODO
operators = [
    ParamScale(gamma_shape, 0.5, Uniform(0, 1), rng, weight=1.0),
    ParamScale(kappa, 0.1, Uniform(0, 1), rng, weight=0.1),
    EpochScale(tree, 0.9, Uniform(0, 1), rng, weight=4.0),
    NarrowExchange(tree, rng, weight=15.0),
    WideExchange(tree, rng, weight=3.0),
    WilsonBalding(tree, rng, weight=3.0),
    NodeSlide(tree, Uniform(0, 1), rng, weight=15.0),
    TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=2.0),
    SubtreePruneRegraft(tree, rng, weight=3),
    ParamScale(birth_rate_y, 0.1, Uniform(0, 1), rng, weight=3),
]

# TODO: frequencies from alignment
sub_model = HKY((0.25, 0.25, 0.25, 0.25), kappa)
clock_model = StrictClock(Real(1.0))
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
            "mutation_rate": mutation_rate,
            "gamma_shape": gamma_shape,
            "kappa": kappa,
            "birth_rate_y": birth_rate_y,
            "population": population,
            "coalescent": priors[-1],
        },
        path="target/b3.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=10_000,
    state=params + [tree],
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    loggers=loggers,
    rng=rng,
)

mcmc.run()

print_operator_stats(mcmc)
