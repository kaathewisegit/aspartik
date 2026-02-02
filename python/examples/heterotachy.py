"""
Based on the [GHOST paper][g], HKY+H4 version.

[g]: https://doi.org/10.1093/sysbio/syz051
"""

from copy import deepcopy
from datetime import datetime

from aspartik.b3 import MCMC, Clock
from aspartik.b3.likelihoods import CPU4Likelihood, HeteroLikelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger, ValueLogger
from aspartik.b3.operators import (
    ClassvecFlip,
    DeltaExchange,
    FixedHeightSubtreePruneRegraft,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import Bound, Distribution, Yule
from aspartik.b3.substitutions import HKY
from aspartik.b3.utils import run_from_cmdline
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

msa = read_msa_from_fasta("data/alignments/electricFish.fasta")

rng = RNG(4)

N = 4

tree = Tree(msa.sequence_names(), rng)
kappas = [Real(2.0) for _ in range(N)]
freqs = [RealVector(0.25, 0.25, 0.25, 0.25) for _ in range(N)]
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
    *(ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1) for kappa in kappas),
    SubtreeLeap(tree, Normal(0, 1), rng, weight=12 * N),
    FixedHeightSubtreePruneRegraft(tree, rng, weight=4 * N),
    ParamScale(birth_rate, 0.75, Uniform(0, 1), rng, weight=3 * N),
    *(DeltaExchange(freq, 0.01, rng, weight=3) for freq in freqs),
    ClassvecFlip(likelihood.class_vector, rng, weight=3),
]

loggers = [
    TreeLogger(tree=tree, path="target/heterotachy.trees", every=1_000),
    PrintLogger(every=10_000),
    ValueLogger(
        {
            "step": lambda: mcmc.current_step,
            "posterior": lambda: mcmc.posterior,
            "prior": lambda: mcmc.prior,
            "likelihood": lambda: mcmc.likelihood.likelihood(),
            "kappas": kappas,
            "birth_rate": birth_rate,
            "clock_rate": clock_rate,
            "frequencies": freqs,
            "tree:height": lambda: tree.height_of(tree.root),
            "tree:length": lambda: tree.total_length(),
            "classes": lambda: likelihood.class_vector.into_list(),
        },
        path="target/heterotachy.log",
        every=1_000,
    ),
]

mcmc = MCMC(
    state=[*kappas, *freqs, birth_rate, clock_rate, tree, likelihood.class_vector],
    priors=priors,
    operators=operators,
    likelihood=likelihood,
    callbacks=loggers,
    rng=rng,
)

run_from_cmdline(mcmc)
