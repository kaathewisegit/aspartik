from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import (
    PrintLogger,
    StateCheckpoint,
    TraceWriter,
)
from aspartik.b3.likelihoods import DNALikelihood
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
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)

population_size = Real(2.0)
frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)
clock_categories = ClassVector(tree.num_nodes, tree.num_edges + 1)

priors = [
    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
    ConstantPopulation(tree, population_size),
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
    Bound(frequencies),
    Bound(rates),
]

operators = [
    DeltaExchange(frequencies, rng, weight=3),
    DeltaExchange(rates, rng, weight=1),
    ClassvecFlip(clock_categories, rng, weight=50),
    ScaleReal(population_size, Uniform(0, 1), rng, weight=3),
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=100),
    FixedHeightSPR(tree, rng, weight=10),
]

likelihood = DNALikelihood(
    msa=msa,
    substitution=GTR(frequencies, rates),
    clock=Clock.Relaxed(clock_categories, LogNormal(1.0, 1.0)),
    tree=tree,
    calculator=Calculator.CPU(),
)

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "population_size": population_size,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "clock_categories": clock_categories,
        },
        "target/relaxed.trace",
        overwrite=True,
        zstd=True,
        every=1_000,
    ),
    StateCheckpoint("target/relaxed.state", every=10_000),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
    optimization_cutoff=100_000,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
