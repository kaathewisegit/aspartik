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
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.b3.utils.skyline import plot_skyline_birthdeath
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Normal, Uniform

msa = read_msa_from_fasta("data/alignments/hcv.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(10, rng)
tree.accept()

become_uninfectious_rate = Real(1)
reproductive_number = RealVector.repeat(2, 20)
sampling_proportion = Real(1)
origin = Real(1000)

frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)

priors = [
    BirthDeathSkyline(
        tree,
        origin,
        become_uninfectious_rate,
        reproductive_number,
        sampling_proportion,
    ),
    Bound(frequencies),
    Bound(rates),
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
]

operators = [
    ScaleReal(origin, Uniform(0, 1), rng, weight=5),
    ScaleReal(become_uninfectious_rate, Uniform(0, 1), rng, weight=2),
    ScaleRealVector(reproductive_number, Uniform(0, 1), rng, weight=10),
    ScaleReal(sampling_proportion, Uniform(0, 1), rng, weight=10),
    #
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
            "origin": origin,
            "become_uninfectious_rate": become_uninfectious_rate,
            "reproductive_number": reproductive_number,
            "sampling_proportion": sampling_proportion,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "prior:bdsky": priors[0],
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
    run_from_cmdline(mcmc, default_length=200_000)

    df = pl.read_ipc("target/bdsky.trace", memory_map=False)
    df = df.slice(df.height // 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Years ago")
    plot_skyline_birthdeath(
        fig,
        ax,
        df["origin"],
        df["reproductive_number"],
        df["become_uninfectious_rate"],
    )
    fig.tight_layout()
    fig.savefig("target/bdsky.png", dpi=300, bbox_inches="tight")
