# WORKING IN PROGRESS

from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import HeteroLikelihood
from aspartik.b3.operators import (
    ClassvecFlip,
    DeltaExchange,
    FixedHeightSPR,
    RootSlide,
    ScaleReal,
    SubtreeLeap,
)
from aspartik.b3.parameters import ClassVector, Real, RealVector, Tree
from aspartik.b3.priors import (
    Bound,
    ConstantPopulation,
    Distribution,
)
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

N = 4

kappas = [Real(2.0) for _ in range(N)]
freqs = [RealVector(0.25, 0.25, 0.25, 0.25) for _ in range(N)]
population_size = Real(1.0)
clock_rates = [Real(0.001) for _ in range(N)]
classes = ClassVector(N, 845)

priors = [
    *(Bound(kappa) for kappa in kappas),
    *(Bound(freq) for freq in freqs),
    Bound(population_size),
    *(Bound(clock_rate) for clock_rate in clock_rates),
    *(Distribution(kappa, LogNormal(1.0, 1.25)) for kappa in kappas),
    *(Distribution(clock_rate, Laplace(0, 0.5)) for clock_rate in clock_rates),
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    ConstantPopulation(tree, population_size),
]


likelihood = HeteroLikelihood(
    msa=msa,
    tree=tree,
    classes=classes,
    substitutions=[HKY(freq, kappa) for freq, kappa in zip(freqs, kappas)],
    clocks=[Clock.Strict(clock_rate) for clock_rate in clock_rates],
    calculator=Calculator.CPU(),
)

operators = [
    *(ScaleReal(kappa, Uniform(0, 1), rng, weight=1) for kappa in kappas),
    *(DeltaExchange(freq, rng, weight=3) for freq in freqs),
    *(
        ScaleReal(clock_rate, Uniform(0, 1), rng, weight=3)
        for clock_rate in clock_rates
    ),
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=12 * N),
    FixedHeightSPR(tree, rng, weight=4 * N),
    ScaleReal(population_size, Uniform(0, 1), rng, weight=3 * N),
    ClassvecFlip(classes, rng, weight=3),
]

loggers = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            # TODO: lists
            "tree": tree,
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
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
