from typing import Literal

from aspartik.b3 import MCMC, Clock, Tree
from aspartik.b3.likelihoods import CPU4Likelihood, CUDALikelihood, Parallel4Likelihood
from aspartik.b3.loggers import PrintLogger
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSubtreePruneRegraft,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, Weights
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth, Yule
from aspartik.b3.substitutions import HKY
from aspartik.data.msa import MSA
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

type Calculator = Literal["cpu", "parallel", "cuda"]


def default(msa: MSA, rng: RNG, kind: Calculator) -> MCMC:
    """
    Default configuration for benchmarking purposes

    Uses HKY with empirical frequencies and an exponential growth coalescent.
    """

    tree = Tree(msa.sequence_names(), rng)

    kappa = Real(2.0)
    population_size = Real(1.0)
    growth_rate = Real(0)
    clock_rate = Real(0.001)
    frequencies = Weights(0.25, 0.25, 0.25, 0.25)
    params = [kappa, population_size, growth_rate, clock_rate, frequencies]

    priors = [
        Bound(kappa),
        Bound(population_size),
        Bound(clock_rate),
        Bound(frequencies),
        Distribution(kappa, LogNormal(1.0, 1.25)),
        Distribution(clock_rate, Laplace(0, 0.5)),
        Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
        Distribution(growth_rate, Laplace(0, 100)),
        ExponentialGrowth(tree, population_size, growth_rate),
    ]

    operators = [
        ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
        ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
        UpDown(Internals(tree), clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
        SubtreeLeap(tree, Normal(0, 1), rng, weight=msa.num_sequences),
        FixedHeightSubtreePruneRegraft(tree, rng, weight=msa.num_sequences / 10),
        ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
        RandomWalk(growth_rate, window=1, rng=rng, weight=3),
        DeltaExchange(frequencies, 0.01, rng, weight=3),
    ]

    match kind:
        case "cpu":
            calculator = CPU4Likelihood
        case "parallel":
            calculator = Parallel4Likelihood
        case "cuda":
            calculator = CUDALikelihood

    likelihood = calculator(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
    )

    mcmc = MCMC(
        state=params + [tree],
        priors=priors,
        operators=operators,
        likelihood=likelihood,
        callbacks=[
            PrintLogger(every=10_000),
        ],
        rng=rng,
    )

    return mcmc
