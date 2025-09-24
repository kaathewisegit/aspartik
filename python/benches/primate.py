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

mutation_rate_noncoding = Real(1.0)
gamma_shape_noncoding = Real(1.0)
kappa_noncoding = Real(2.0)
mutation_rate_1stpos = Real(1.0)
gamma_shape_1stpos = Real(1.0)
kappa_1stpos = Real(2.0)
mutation_rate_2ndpos = Real(1.0)
gamma_shape_2ndpos = Real(1.0)
kappa_2ndpos = Real(2.0)
mutation_rate_3rdpos = Real(1.0)
gamma_shape_3rdpos = Real(1.0)
kappa_3rdpos = Real(2.0)

population = Real(1.0)
birth_rate_y = Real(1.0)
clock_rate = Real(1.0)

params = [
    mutation_rate_noncoding,
    gamma_shape_noncoding,
    kappa_noncoding,
    mutation_rate_1stpos,
    gamma_shape_1stpos,
    kappa_1stpos,
    mutation_rate_2ndpos,
    gamma_shape_2ndpos,
    kappa_2ndpos,
    mutation_rate_3rdpos,
    gamma_shape_3rdpos,
    kappa_3rdpos,
    birth_rate_y,
    clock_rate,
]


homo, pan = tree.leaf_by_name("Homo_sapiens"), tree.leaf_by_name("Pan_troglodytes")
assert homo != None
assert pan != None

priors = [
    Yule(tree, birth_rate_y),
    Distribution(birth_rate_y, Gamma(0.001, 1 / 1000.0)),
    Distribution(gamma_shape_noncoding, Exp(1.0)),
    Distribution(gamma_shape_1stpos, Exp(1.0)),
    Distribution(gamma_shape_2ndpos, Exp(1.0)),
    Distribution(gamma_shape_3rdpos, Exp(1.0)),
    Distribution(kappa_noncoding, LogNormal(1.0, 1.25)),
    Distribution(kappa_1stpos, LogNormal(1.0, 1.25)),
    Distribution(kappa_2ndpos, LogNormal(1.0, 1.25)),
    Distribution(kappa_3rdpos, LogNormal(1.0, 1.25)),
    ConstantPopulation(tree, population),
    Monophyly(tree, [homo, pan]),
]

# TODO
operators = [
    ParamScale(gamma_shape_noncoding, 0.5, Uniform(0, 1), rng, weight=1.0),
    ParamScale(kappa_noncoding, 0.1, Uniform(0, 1), rng, weight=0.1),
    ParamScale(kappa_1stpos, 0.1, Uniform(0, 1), rng, weight=0.1),
    ParamScale(kappa_2ndpos, 0.1, Uniform(0, 1), rng, weight=0.1),
    ParamScale(kappa_3rdpos, 0.1, Uniform(0, 1), rng, weight=0.1),
    EpochScale(tree, 0.9, Uniform(0, 1), rng, weight=4.0),
    NarrowExchange(tree, rng, weight=15.0),
    WideExchange(tree, rng, weight=3.0),
    WilsonBalding(tree, rng, weight=3.0),
    NodeSlide(tree, Uniform(0, 1), rng, weight=15.0),
    TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=2.0),
    DeltaExchange(
        params=[
            mutation_rate_noncoding,
            mutation_rate_1stpos,
            mutation_rate_2ndpos,
            mutation_rate_3rdpos,
        ],
        weights=[205, 231, 231, 231],
        factor=0.75,
        rng=rng,
        weight=2.0,
    ),
    SubtreePruneRegraft(tree, rng, weight=3),
    ParamScale(birth_rate_y, 0.1, Uniform(0, 1), rng, weight=3),
    ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
]

# TODO: frequencies from alignment
sub_model = HKY((0.25, 0.25, 0.25, 0.25), kappa_noncoding)
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
            "mutation_rate_noncoding": mutation_rate_noncoding,
            "gamma_shape_noncoding": gamma_shape_noncoding,
            "kappa_noncoding": kappa_noncoding,
            "mutation_rate_1stpos": mutation_rate_1stpos,
            "gamma_shape_1stpos": gamma_shape_1stpos,
            "kappa_1stpos": kappa_1stpos,
            "mutation_rate_2ndpos": mutation_rate_2ndpos,
            "gamma_shape_2ndpos": gamma_shape_2ndpos,
            "kappa_2ndpos": kappa_2ndpos,
            "mutation_rate_3rdpos": mutation_rate_3rdpos,
            "gamma_shape_3rdpos": gamma_shape_3rdpos,
            "kappa_3rdpos": kappa_3rdpos,
            "birth_rate_y": birth_rate_y,
            "clock_rate": clock_rate,
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
