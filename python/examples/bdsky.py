import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import DNALikelihood
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSPR,
    RootSlide,
    ScaleReal,
    ScaleRealVector,
    SubtreeLeap,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import (
    BirthDeathSkyline,
    Bound,
    Distribution,
    SymmetricDirichlet,
    VectorDistribution,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.b3.utils.skyline import plot_skyline_birthdeath
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/hcv.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(0.1, rng)
tree.accept()

NUM_INTERVALS = 10
times = RealVector(*[20.0 * i for i in range(NUM_INTERVALS)])
birth_rates = RealVector.repeat(1.0, NUM_INTERVALS)
death_rates = RealVector.repeat(0.5, NUM_INTERVALS)
sampling_rates = RealVector.repeat(0.25, NUM_INTERVALS)
origin = Real(150.0)

frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)

bdsky = BirthDeathSkyline(
    tree,
    times,
    birth_rates,
    death_rates,
    sampling_rates,
    origin,
    relative_death=False,
    times_start_from_origin=False,
    condition_on_survival=True,
)

priors = [
    Bound(birth_rates),
    Bound(death_rates),
    Bound(origin),
    Bound(frequencies),
    Bound(rates),
    VectorDistribution(birth_rates, LogNormal(0.0, 1.25)),
    VectorDistribution(death_rates, LogNormal(0.0, 1.25)),
    Distribution(origin, LogNormal(5.0, 0.5)),
    bdsky,
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
]

operators = [
    ScaleRealVector(birth_rates, Uniform(0, 1), rng, weight=15),
    ScaleRealVector(death_rates, Uniform(0, 1), rng, weight=15),
    ScaleReal(origin, Uniform(0, 1), rng, weight=5),
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
            "birth_rates": birth_rates,
            "death_rates": death_rates,
            "sampling_rates": sampling_rates,
            "origin": origin,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "prior:bdsky": bdsky,
        },
        path="target/bdsky.trace",
        every=10_000,
        overwrite=True,
        zstd=True,
    ),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc, default_length=1_000_000)

    df = pl.read_ipc("target/bdsky.trace", memory_map=False)
    offset = int(len(df) * 0.1)
    df = df.slice(offset)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Years ago")
    ax.set_ylabel("R_e")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.invert_xaxis()
    plot_skyline_birthdeath(
        fig,
        ax,
        [times[i] for i in range(len(times))],
        df["origin"],
        birth_rates=df["birth_rates"],
        death_rates=df["death_rates"],
        trees=df["tree"],
        sequence_names=msa.sequence_names(),
        times_start_from_origin=False,
        mode="hpd",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("target/bdsky.png", dpi=300, bbox_inches="tight")
