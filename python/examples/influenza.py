"""
Based on BEAST's ["Estimating rates and dates" workshop
tutorial][l].

[l]: https://beast.community/workshop_influenza_phylodynamics
"""

from datetime import datetime

from aspartik.b3 import MCMC, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.likelihoods import CPU4Likelihood, Parallel4Likelihood
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
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth, Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform


def make_mcmc(fasta_path: str):
    msa = read_msa_from_fasta(fasta_path)

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
        SubtreeLeap(tree, Normal(0, 1), rng, weight=50),
        SubtreePruneRegraft(tree, rng, weight=5),
        ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
        RandomWalk(growth_rate, window=1, rng=rng, weight=3),
        DeltaExchange(frequencies, 0.01, rng, weight=3),
    ]

    likelihood = Parallel4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=StrictClock(clock_rate),
        tree=tree,
        num_internal_threads=3,
    )

    loggers = [
        TreeLogger(tree=tree, path="target/influenza.trees", every=1_000),
        PrintLogger(every=10_000),
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
            path="target/influenza.log",
            every=1_000,
            zstd=True,
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

    return mcmc


def run(fasta_path: str):
    run_from_cmdline(make_mcmc(fasta_path))


if __name__ == "__main__":
    run("data/alignments/influenza.fasta")
