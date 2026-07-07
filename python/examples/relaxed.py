from aspartik.b3 import MCMC, Calculator, Clock
from aspartik.b3.callbacks import (
    PrintLogger,
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
from aspartik.stats.distributions import DynLogNormal, Exp, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)
tree = Tree(msa.sequence_names(), rng)
tree.set_random_heights(10, rng)
tree.accept()

frequencies = RealVector.repeat(0.25, 4)
rates = RealVector.repeat(1, 6)
clock_categories = ClassVector(tree.num_nodes, tree.num_edges + 1)
clock_mean = Real(1)
clock_scale = Real(1 / 3)

priors = [
    SymmetricDirichlet(frequencies, 1),
    SymmetricDirichlet(rates, 6),
    Bound(frequencies),
    Bound(rates),
    Distribution(clock_mean, Normal(0, 3)),
    Distribution(clock_scale, Exp(3)),
    #
    Distribution(Root(tree), Normal(284.1, 0.5)),
]

operators = [
    DeltaExchange(frequencies, rng, weight=3),
    DeltaExchange(rates, rng, weight=1),
    ClassvecFlip(clock_categories, rng, weight=50),
    RootSlide(tree, Uniform(0, 1), rng, weight=3),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=100),
    FixedHeightSPR(tree, rng, weight=10),
    ScaleReal(clock_mean, Uniform(0, 1), rng, weight=10),
    ScaleReal(clock_scale, Uniform(0, 1), rng, weight=10),
]

clock = Clock.Relaxed(clock_categories, DynLogNormal(clock_mean, clock_scale))
likelihood = DNALikelihood(
    msa=msa,
    substitution=GTR(frequencies, rates),
    clock=clock,
    tree=tree,
    calculator=Calculator.CPU(),
)

callbacks = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            "frequencies": frequencies,
            "rates": rates,
            "tree": tree,
            "clock_mean": clock_mean,
            "clock_scale": clock_scale,
            "clock_categories": clock_categories,
        },
        "target/relaxed.trace",
        overwrite=True,
        zstd=True,
        every=10_000,
    ),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=callbacks,
    rng=rng,
    optimization_cutoff=5_000_000,
)


if __name__ == "__main__":
    run_from_cmdline(mcmc)
    print(tree.to_newick(clock=clock))
