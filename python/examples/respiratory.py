"""
Based on BEAST's "Phylodynamic inference of respiratory viruses" workshop
tutorial:

<https://beast.community/workshop_respiratory_virus_phylodynamics>
"""

from collections import deque
from datetime import datetime
from pathlib import Path

from aspartik.b3 import MCMC, Internal, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.likelihoods import CUDALikelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    DeltaExchange,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    SubtreePruneRegraft,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, Weights
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import print_operator_stats, print_operator_timings
from aspartik.data import newick
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/respiratory.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

times = []
for name in msa.sequence_names():
    node = tree.leaf_by_name(name)
    assert node is not None

    time_s = name.split("|")[-1]
    time = datetime.strptime(time_s, "%Y-%m-%d")
    times.append(time)

max_time = max(times)

for leaf, time in zip(tree.leaves(), times):
    height = (max_time - time).days / 365
    tree.set_height(leaf, height)

tree.set_random_heights(0.001, rng)
tree.accept()


kappa = Real(2.0)
population_size = Real(1.0)
growth_rate = Real(0.0)
clock_rate = Real(0.001)
frequencies = Weights(0.25, 0.25, 0.25, 0.25)
params = [kappa, population_size, growth_rate, clock_rate]

priors = [
    Bound(kappa),
    Bound(population_size),
    Bound(clock_rate),
    Bound(frequencies),
    Distribution(kappa, LogNormal(1.0, 1.25)),
    Distribution(clock_rate, Laplace(0, 0.5)),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    Distribution(growth_rate, Laplace(0.0, 100.0)),
    ExponentialGrowth(tree, population_size, growth_rate),
]

operators = [
    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
    ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
    UpDown(Internals(tree), clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=1000),
    SubtreePruneRegraft(tree, rng, weight=100),
    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
    RandomWalk(growth_rate, window=1.0, rng=rng, weight=3),
]


likelihood = CUDALikelihood(
    msa=msa,
    substitution=HKY(frequencies, kappa),
    clock=StrictClock(clock_rate),
    tree=tree,
    cuda_device=1,
)

loggers = [
    TreeLogger(tree=tree, path="target/respiratory.trees", every=1_000),
    PrintLogger(every=1_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "joint": lambda: mcmc.prior + mcmc.likelihood.likelihood(),
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.likelihood.likelihood(),
            "kappa": kappa,
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "tree:height": lambda: tree.height_of(tree.root),
            "tree:length": lambda: tree.total_length(),
            "prior:kappa": priors[4],
            "prior:clock_rate": priors[5],
            "prior:population_size": priors[6],
            "prior:growth_rate": priors[7],
            "prior:coalescent": priors[8],
        },
        path="target/respiratory.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    burnin=0,
    length=1_000_000_000,
    state=params + [tree],
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)


try:
    mcmc.run()
except Exception as e:
    print(e)

print_operator_stats(mcmc)
print()
print_operator_timings(mcmc)
