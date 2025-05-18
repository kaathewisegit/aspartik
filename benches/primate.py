from aspartik import b3
from aspartik.b3 import Tree, State, Parameter, Likelihood
from aspartik.b3.loggers import TreeLogger
from aspartik.b3.operators import (
    TreeScale,
    NarrowExchange,
    WideExchange,
    NodeSlide,
    DeltaExchange,
)
from aspartik.b3.priors import Distribution
from aspartik.b3.substitutions import JC
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


state = State(tree, params)

# TODO: limit priors
priors = [
    # TODO: Yule model
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
    NarrowExchange(tree, rng, weight=25.0),
    WideExchange(tree, rng, weight=25.0),
    NodeSlide(tree, Uniform(0, 1), rng, weight=48.0),
    TreeScale(tree, 0.1, Uniform(0, 1), rng, weight=2.0),
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
]

# TODO: HKY substitution
likelihood = Likelihood(
    data="crates/b3/data/primate-mdna.fasta", substitution=JC(), tree=tree
)

loggers = [
    TreeLogger(tree=tree, path="b3.trees", every=1_000),
]

b3.run(100_000, state, priors, operators, likelihood, loggers, rng)
