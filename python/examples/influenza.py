"""
Based on BEAST's ["Estimating rates and dates" workshop
tutorial][l].

[l]: https://beast.community/workshop_influenza_phylodynamics
"""

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import PrintLogger, StateCheckpoint, TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSPR,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import (
    Bound,
    Distribution,
    ExponentialGrowth,
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/H1N1pdm_2009.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

times = []
for name in msa.sequence_names():
    node = tree.leaf_by_name(name)
    assert node is not None

    time = name.split("_")[-1]
    times.append(float(time))

max_time = max(times)

for leaf, time in zip(tree.leaves(), times):
    height = max_time - time
    tree.set_height(leaf, height)

tree.set_random_heights(0.1, rng)
tree.accept()

population_size = Real(1.0)
growth_rate = Real(0)
clock_rate = Real(0.001)
frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
rates = RealVector(1, 1, 1, 1, 1, 1)

params = [population_size, growth_rate, clock_rate, frequencies, tree]

priors = [
    Bound(population_size),
    Bound(clock_rate),
    Bound(frequencies),
    Distribution(clock_rate, Laplace(0, 0.5)),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    Distribution(growth_rate, Laplace(0, 100)),
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
    ExponentialGrowth(tree, population_size, growth_rate),
]

operators = [
    ParamScale(clock_rate, Uniform(0, 1), rng, weight=3),
    UpDown(Internals(tree), clock_rate, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=50),
    FixedHeightSPR(tree, rng, weight=5),
    ParamScale(population_size, Uniform(0, 1), rng, weight=3),
    RandomWalk(growth_rate, window=10, rng=rng, weight=3),
    DeltaExchange(frequencies, rng, weight=3),
    DeltaExchange(rates, rng, weight=1),
]

likelihood = CPU4Likelihood(
    msa=msa,
    substitution=GTR(frequencies, rates),
    clock=Clock.Strict(clock_rate),
    tree=tree,
)

loggers = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "prior:clock_rate": priors[4],
            "prior:population_size": priors[5],
            "prior:growth_rate": priors[6],
            "prior:coalescent": priors[8],
        },
        path="target/influenza.trace",
        every=100,
        overwrite=True,
        zstd=True,
    ),
    StateCheckpoint("target/influenza.state", every=100_000),
]

mcmc = MCMC(
    state=params,
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
