# WORK IN PROGRESS

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import HeteroLikelihood
from aspartik.b3.operators import (
    ClassvecFlip,
    DeltaExchange,
    FixedHeightSPR,
    RootSlide,
    SubtreeLeap,
)
from aspartik.b3.parameters import ClassVector, Real, RealVector, Root, Tree
from aspartik.b3.priors import (
    Bound,
    Distribution,
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import DynLogNormal, Laplace, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(10, rng)

N = 4

rates = [RealVector.repeat(1, 6) for _ in range(N)]
frequencies = [RealVector.repeat(0.25, 4) for _ in range(N)]
categories = ClassVector(N, 845)
clock_categories = [ClassVector(30, tree.num_nodes) for _ in range(N)]

priors = [
    Distribution(Root(tree), Laplace(284.1, 5)),
    *(Bound(freqs) for freqs in frequencies),
    *(Bound(rate) for rate in rates),
    *(SymmetricDirichlet(rate, 6) for rate in rates),
    *(SymmetricDirichlet(freqs, 1) for freqs in frequencies),
]

likelihood = HeteroLikelihood(
    msa=msa,
    tree=tree,
    categories=categories,
    substitutions=[GTR(rate, freqs) for rate, freqs in zip(frequencies, rates)],
    clocks=[
        Clock.Relaxed(clock_cats, DynLogNormal(Real(1), Real(1 / 3)))
        for clock_cats in clock_categories
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
]

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            **{
                f"clock_categories:{i}": clock_cats
                for i, clock_cats in enumerate(clock_categories)
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
    optimization_cutoff=1_000_000,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
