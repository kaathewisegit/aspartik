"""
Based on the [GHOST paper][g], HKY+H4 version.

[g]: https://doi.org/10.1093/sysbio/syz051
"""

from copy import deepcopy
from datetime import datetime

from aspartik.b3 import MCMC, Clock, Tree
from aspartik.b3.likelihoods import CPU4Likelihood, WeightedLikelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSubtreePruneRegraft,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, Weights
from aspartik.b3.priors import Bound, ConstantPopulation, Distribution
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)


N = 2


def repeat(v, count: int):
    return [deepcopy(v) for _ in range(count)]


kappas = repeat(Real(2.0), N)
freqs = repeat(Weights(0.25, 0.25, 0.25, 0.25), N)
population_size = Real(1.0)
clock_rates = repeat(Real(0.001), N)
likelihood_weights = Weights(*repeat(1 / N, N))

params = [
    *kappas,
    *freqs,
    population_size,
    *clock_rates,
    likelihood_weights,
]

priors = [
    *(Bound(kappa) for kappa in kappas),
    *(Bound(freq) for freq in freqs),
    Bound(population_size),
    *(Bound(clock_rate) for clock_rate in clock_rates),
    Bound(likelihood_weights),
    *(Distribution(kappa, LogNormal(1.0, 1.25)) for kappa in kappas),
    *(Distribution(clock_rate, Laplace(0, 0.5)) for clock_rate in clock_rates),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    ConstantPopulation(tree, population_size),
]


operators = [
    *(ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1) for kappa in kappas),
    *(
        ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3)
        for clock_rate in clock_rates
    ),
    *(
        UpDown(Internals(tree), clock_rate, 0.75, Uniform(0, 1), rng, weight=3)
        for clock_rate in clock_rates
    ),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=12 * N),
    FixedHeightSubtreePruneRegraft(tree, rng, weight=4 * N),
    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3 * N),
    *(DeltaExchange(freq, 0.01, rng, weight=3) for freq in freqs),
    DeltaExchange(likelihood_weights, 0.01, rng, weight=3 * N),
]

likelihood = WeightedLikelihood(
    weights=likelihood_weights,
    likelihoods=[
        CPU4Likelihood(
            msa=msa,
            substitution=HKY(freq, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
        for kappa, freq, clock_rate in zip(kappas, freqs, clock_rates)
    ],
)

loggers = [
    TreeLogger(tree=tree, path="target/heterotachy.trees", every=1_000),
    PrintLogger(every=10_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "joint": lambda: mcmc.prior + mcmc.likelihood.likelihood(),
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.likelihood.likelihood(),
            "kappas": kappas,
            "population_size": population_size,
            "clock_rates": clock_rates,
            "frequencies": freqs,
            "likelihood_weights": likelihood_weights,
            "tree:height": lambda: tree.height_of(tree.root),
            "tree:length": lambda: tree.total_length(),
        },
        path="target/heterotachy.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    state=params + [tree],
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)

run_from_cmdline(mcmc)
