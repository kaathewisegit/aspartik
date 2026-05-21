"""
Based on the [GHOST paper][g], HKY+H4 version.

[g]: https://doi.org/10.1093/sysbio/syz051
"""

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import PrintLogger, TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood, HeteroLikelihood
from aspartik.b3.operators import (
    ClassvecFlip,
    DeltaExchange,
    FixedHeightSPR,
    ScaleReal,
    SubtreeLeap,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import Bound, Distribution, Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)

N = 4

tree = Tree(msa.sequence_names(), rng)
kappas = [Real(2.0) for _ in range(N)]
freqs = [RealVector.repeat(0.25, 4) for _ in range(N)]
birth_rate = Real(1.0)
clock_rate = Real(0.001)  # TODO

priors = [
    *(Bound(kappa) for kappa in kappas),
    *(Bound(freq) for freq in freqs),
    Bound(birth_rate),
    *(Distribution(kappa, LogNormal(1.0, 1.25)) for kappa in kappas),
    Distribution(birth_rate, Gamma(0.001, 1 / 1000.0)),
    Yule(tree, birth_rate),
]


likelihood = HeteroLikelihood(
    likelihoods=[
        CPU4Likelihood(
            msa=msa,
            substitution=HKY(freq, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
        for kappa, freq in zip(kappas, freqs)
    ],
)

operators = [
    *(ScaleReal(kappa, Uniform(0, 1), rng, weight=1) for kappa in kappas),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=12 * N),
    FixedHeightSPR(tree, rng, weight=4 * N),
    ScaleReal(birth_rate, Uniform(0, 1), rng, weight=3 * N),
    *(DeltaExchange(freq, rng, weight=3) for freq in freqs),
    ClassvecFlip(likelihood.class_vector, rng, weight=3),
]

loggers = [
    PrintLogger(every=10_000),
    TraceWriter(
        {
            **{f"m{i}:kappa": kappa for i, kappa in enumerate(kappas)},
            **{f"m{i}:frequencies": freq for i, freq in enumerate(freqs)},
            "birth_rate": birth_rate,
            "clock_rate": clock_rate,
            "tree": tree,
            "classes": likelihood.class_vector,
        },
        path="target/heterotachy.trace",
        overwrite=True,
        every=1_000,
    ),
]

mcmc = MCMC(
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)

run_from_cmdline(mcmc)
