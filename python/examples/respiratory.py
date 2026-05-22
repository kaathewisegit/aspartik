"""
Based on BEAST's "Phylodynamic inference of respiratory viruses" workshop
tutorial:

<https://beast.community/workshop_respiratory_virus_phylodynamics>
"""

from datetime import datetime

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, Timer, TraceWriter
from aspartik.b3.likelihoods import DNALikelihood
from aspartik.b3.operators import (
    FixedHeightSPR,
    RandomWalk,
    ScaleReal,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/b.1.1.7.fasta")

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
frequencies = RealVector.repeat(0.25, 4)

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
    ScaleReal(kappa, Uniform(0, 1), rng, weight=1),
    ScaleReal(clock_rate, Uniform(0, 1), rng, weight=3),
    UpDown(tree, clock_rate, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=1000),
    FixedHeightSPR(tree, rng, weight=100),
    ScaleReal(population_size, Uniform(0, 1), rng, weight=3),
    RandomWalk(growth_rate, window=10, rng=rng, weight=3),
]

try:
    likelihood = DNALikelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CUDA(),
    )
except Exception as e:
    print(f"failed to create CUDALikelihood: {e}")
    likelihood = DNALikelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        calculator=Calculator.CPU(),
    )

loggers = [
    PrintLogger(every=1_000),
    TraceWriter(
        {
            "kappa": kappa,
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "tree": tree,
            "prior:kappa": priors[4],
            "prior:clock_rate": priors[5],
            "prior:population_size": priors[6],
            "prior:growth_rate": priors[7],
            "prior:coalescent": priors[8],
        },
        path="target/respiratory.trace",
        every=1_000,
        overwrite=True,
        zstd=True,
    ),
    Timer(),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc, default_length=10_000)
