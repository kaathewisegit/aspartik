import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, StateCheckpoint, TraceWriter
from aspartik.b3.likelihoods import DNALikelihood
from aspartik.b3.operators import (
    DeltaExchange,
    DeltaExchangeInt,
    FixedHeightSPR,
    RootSlide,
    ScaleRealVector,
    SubtreeLeap,
)
from aspartik.b3.parameters import IntVector, Real, RealVector, Tree
from aspartik.b3.priors import (
    BayesianSkyline,
    Bound,
    BoundInt,
    MarkovChainDistribution,
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.b3.utils.skyline import plot_skyline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Normal, Uniform

msa = read_msa_from_fasta("data/alignments/hcv.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(0.1, rng)
tree.accept()

NUM_GROUPS = 4
population_sizes = RealVector.repeat(1.0, NUM_GROUPS)
group_sizes = IntVector.repeat(0, NUM_GROUPS)  # will be overwritten by BayesianSkyline

frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)


priors = [
    MarkovChainDistribution(population_sizes),
    BayesianSkyline(tree, population_sizes, group_sizes),
    Bound(population_sizes),
    BoundInt(group_sizes),
    Bound(frequencies),
    Bound(rates),
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
]

operators = [
    ScaleRealVector(population_sizes, Uniform(0, 1), rng, weight=15),
    DeltaExchangeInt(group_sizes, rng, weight=6),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=50),
    FixedHeightSPR(tree, rng, weight=5),
    RootSlide(tree, Uniform(0, 1), rng, weight=5),
    DeltaExchange(frequencies, rng, weight=3),
    DeltaExchange(rates, rng, 1.0, weight=1),
]

likelihood = DNALikelihood(
    msa=msa,
    substitution=GTR(frequencies, rates),
    clock=Clock.Strict(Real(7.9e-4)),
    tree=tree,
    calculator=Calculator.CPU(),
)

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "population_sizes": population_sizes,
            "group_sizes": group_sizes,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "prior:coalescent": priors[1],
        },
        path="target/skyline.trace",
        every=10_000,
        overwrite=True,
        zstd=True,
    ),
    StateCheckpoint("target/skyline.state", every=500_000),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc, default_length=200_000)

    offset = mcmc.current_step // (10_000 * 10)
    sequence_names = msa.sequence_names()
    df = pl.read_ipc("target/skyline.trace", memory_map=False).slice(offset)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")
    ax.set_xlabel("Years ago")
    ax.set_ylabel("Population size")
    ax.invert_xaxis()
    plot_skyline(fig, ax, df, "population_sizes", sequence_names, mode="hpd")
    ax.legend()
    fig.tight_layout()
    fig.savefig("target/skyline.png", dpi=300, bbox_inches="tight")
