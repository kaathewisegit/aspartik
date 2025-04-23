import b3
from b3 import Tree, State, Parameter, Likelihood
from b3.loggers import TreeLogger
from b3.operators import TreeScale, NarrowExchange, WideExchange, TreeSlide
from b3.priors import Bound, Distribution
from b3.substitutions import JC
from rng import Rng
from stats.distributions import Uniform, Gamma, Poisson

rng = Rng(4)
tree = Tree(100, rng)
tree.verify()

params = [
    Parameter.Real(0.5),
    Parameter.Integer(0, 1, 2, 3),
    Parameter.Boolean(True, False),
]

state = State(tree, params, rng)

priors = [
    Bound(params[0], lower=0, upper=1),
    Distribution(params[0], Gamma(2, 1)),
    Distribution(params[1], Poisson(1)),
]

operators = [
    NarrowExchange(weight=25.0),
    WideExchange(weight=25.0),
    TreeSlide(Uniform(0, 1), weight=48.0),
    TreeScale(0.1, Uniform(0, 1), weight=2.0),
]

likelihood = Likelihood(data="data/100.fasta", substitution=JC())

loggers = [
    TreeLogger(path="b3.trees", every=1_000),
]

b3.run(10_000, state, priors, operators, likelihood, loggers)
