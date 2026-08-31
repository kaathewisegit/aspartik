import polars as pl

from math import inf

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import HeteroLikelihood
from aspartik.b3.operators import (
    ClassvecFlip,
    DeltaExchange,
    FixedHeightSPR,
    RandomWalk,
    RootSlide,
    ScaleReal,
    SubtreeLeap,
)
from aspartik.b3.parameters import ClassVector, Real, RealVector, Root, Tree
from aspartik.b3.priors import Bound, Distribution, Monophyly, SymmetricDirichlet
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.b3.utils.heterotachy import pattern_probabilities, site_probabilities
from aspartik.data.msa import MSA
from aspartik.distributions import Exponential, Laplace, LogNormal, Normal, Uniform
from aspartik.rng import RNG

msa = MSA.from_fasta_file("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(20, rng)
tree.accept()
leaves = list(tree.leaves())

N = 4

rates = [RealVector.repeat(1, 6) for _ in range(N)]
frequencies = [RealVector.repeat(0.25, 4) for _ in range(N)]
categories = ClassVector(N, 845)
clock_categories = [ClassVector(15, tree.num_nodes) for _ in range(N)]
clock_locations = [Real(1e-5) for _ in range(N)]
clock_scales = [Real(1 / 3) for _ in range(N)]

priors = [
    Distribution(Root(tree), Laplace(284.1, 5)),
    # Osteoglossocephala: Silver Arowana, Clown Knifefish, Elephantnose
    Monophyly(tree, leaves[:3]),
    # Clupeocephala: everyone else
    Monophyly(tree, list(tree.leaves())[3:]),
    #
    *(Bound(freqs) for freqs in frequencies),
    *(Bound(rate) for rate in rates),
    *(SymmetricDirichlet(rate, 6) for rate in rates),
    *(SymmetricDirichlet(freqs, 1) for freqs in frequencies),
    *(
        Distribution(clock_location, Normal(0, 10))
        for clock_location in clock_locations
    ),
    *(Distribution(clock_scale, Exponential(3)) for clock_scale in clock_scales),
]

likelihood = HeteroLikelihood(
    msa=msa,
    tree=tree,
    categories=categories,
    substitutions=[GTR(freqs, rate) for freqs, rate in zip(frequencies, rates)],
    clocks=[
        Clock.Relaxed(clock_cats, LogNormal(clock_location, clock_scale))
        for clock_cats, clock_location, clock_scale in zip(
            clock_categories, clock_locations, clock_scales
        )
    ],
    calculator=Calculator.CPU(),
)

operators = [
    *(DeltaExchange(rate, rng, weight=1) for rate in rates),
    *(DeltaExchange(freqs, rng, weight=3) for freqs in frequencies),
    *(ClassvecFlip(clock_cats, rng, weight=50) for clock_cats in clock_categories),
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=12 * N),
    FixedHeightSPR(tree, rng, weight=4 * N),
    ClassvecFlip(categories, rng, weight=100),
    *(
        RandomWalk(clock_location, lower=-inf, window=10, rng=rng, weight=10)
        for clock_location in clock_locations
    ),
    *(
        ScaleReal(clock_scale, Uniform(0, 1), rng, weight=10)
        for clock_scale in clock_scales
    ),
]

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            **{
                f"clock_categories:{i}": clock_cats
                for i, clock_cats in enumerate(clock_categories)
            },
            **{
                f"clock_location:{i}": clock_location
                for i, clock_location in enumerate(clock_locations)
            },
            **{
                f"clock_scale:{i}": clock_scale
                for i, clock_scale in enumerate(clock_scales)
            },
            **{f"rates:{i}": rate for i, rate in enumerate(rates)},
            **{f"frequencies:{i}": freqs for i, freqs in enumerate(frequencies)},
            "categories": categories,
            "tree": tree,
            "prior:root": priors[0],
        },
        path="target/heterotachy.trace",
        every=10_000,
        overwrite=True,
        zstd=True,
    ),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    optimization_cutoff=5_000_000,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)

    df = pl.read_ipc("target/heterotachy.trace", memory_map=False)
    df = df.slice(df.height // 2)
    patterns = pattern_probabilities(df["categories"])
    sites = site_probabilities(patterns, likelihood.sites_to_patterns)
    sites.write_csv("target/heterotachy.siteprobs", separator="\t")
    patterns.write_csv("target/heterotachy.patternprobs", separator="\t")
