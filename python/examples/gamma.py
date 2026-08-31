from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import GammaLikelihood
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSPR,
    RandomWalk,
    RootSlide,
    ScaleReal,
    SubtreeLeap,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import (
    Bound,
    Distribution,
    ExponentialGrowth,
    SymmetricDirichlet,
)
from aspartik.b3.substitutions import GTR
from aspartik.b3.utils import run_from_cmdline
from aspartik.data.msa import MSA
from aspartik.distributions import Exponential, Gamma, Laplace, Normal, Uniform
from aspartik.rng import RNG

msa = MSA.from_fasta_file("data/alignments/H1N1pdm_2009.fasta")

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

alpha = Real(1.0)
population_size = Real(1.0)
growth_rate = Real(0)
clock_rate = Real(0.001)
frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)

priors = [
    Distribution(alpha, Exponential(0.5)),
    Distribution(clock_rate, Laplace(0, 0.5)),
    Distribution(population_size, Gamma(0.001, 1000.0)),
    Distribution(growth_rate, Laplace(0, 100)),
    ExponentialGrowth(tree, population_size, growth_rate),
    Bound(alpha, lower=0.1, upper=100),
    Bound(population_size),
    Bound(clock_rate),
    Bound(frequencies),
    Bound(rates),
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
]

operators = [
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=100),
    FixedHeightSPR(tree, rng, weight=10),
    ScaleReal(alpha, Uniform(0, 1), rng, weight=3),
    ScaleReal(clock_rate, Uniform(0, 1), rng, weight=3),
    ScaleReal(population_size, Uniform(0, 1), rng, weight=3),
    RandomWalk(growth_rate, window=10, rng=rng, weight=3),
    DeltaExchange(frequencies, rng, weight=3),
    DeltaExchange(rates, rng, weight=1),
]

likelihood = GammaLikelihood(
    msa=msa,
    substitution=GTR(frequencies, rates),
    num_categories=4,
    alpha=alpha,
    clock=Clock.Strict(clock_rate),
    tree=tree,
    calculator=Calculator.CPU(),
)

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "alpha": alpha,
            "population_size": population_size,
            "growth_rate": growth_rate,
            "clock_rate": clock_rate,
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
        },
        path="target/gamma.trace",
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
    run_from_cmdline(mcmc)
