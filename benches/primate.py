from aspartik.b3 import MCMC, Tree, Parameter, Likelihood
from aspartik.b3.loggers import TreeLogger
from aspartik.b3.operators import (
    ParamScale,
    EpochScale,
    TreeScale,
    RootScale,
    NarrowExchange,
    WideExchange,
    NodeSlide,
    DeltaExchange,
    WilsonBalding,
)
from aspartik.b3.priors import Distribution, Yule
from aspartik.b3.substitutions import HKY
from aspartik.stats.distributions import Gamma, Uniform, Exp, LogNormal
from aspartik.rng import Rng

rng = Rng(4)
tree = Tree(12, rng)

mutation_rate_noncoding = Parameter.Real(1.0)
gamma_shape_noncoding = Parameter.Real(1.0)
kappa_noncoding = Parameter.Real(2.0)
mutation_rate_1stpos = Parameter.Real(1.0)
gamma_shape_1stpos = Parameter.Real(1.0)
kappa_1stpos = Parameter.Real(2.0)
mutation_rate_2ndpos = Parameter.Real(1.0)
gamma_shape_2ndpos = Parameter.Real(1.0)
kappa_2ndpos = Parameter.Real(2.0)
mutation_rate_3rdpos = Parameter.Real(1.0)
gamma_shape_3rdpos = Parameter.Real(1.0)
kappa_3rdpos = Parameter.Real(2.0)

birth_rate_y = Parameter.Real(1.0)
clock_rate = Parameter.Real(1.0)

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


# TODO: limit priors
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
    # TODO: MRCA
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
    RootScale(tree, 0.9, Uniform(0, 1), rng, weight=3.0),
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
    ParamScale(birth_rate_y, 0.1, Uniform(0, 1), rng, weight=3),
    ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
]

# TODO: frequencies from alignment
model = HKY((0.25, 0.25, 0.25, 0.25), kappa_noncoding)
likelihood = Likelihood(
    data="crates/b3/data/primate-mdna.fasta", substitution=model, tree=tree
)

loggers = [
    TreeLogger(tree=tree, path="b3.trees", every=1_000),
]

mcmc = MCMC(
    burnin=0,
    length=5_000_000,
    trees=[tree],
    params=params,
    priors=priors,
    operators=operators,
    likelihoods=[likelihood],
    loggers=loggers,
    rng=rng,
)

mcmc.run()
