from typing import Literal, Optional

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.loggers import PrintLogger, TreeLogger
from aspartik.b3.operators import (
    BeastNarrowExchange,
    BeastWideExchange,
    DeltaExchange,
    FixedHeightSPR,
    NodeSlide,
    ParamScale,
    RootSlide,
    SubtreeLeap,
    SubtreeSlide,
    TreeScale,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import ConstantPopulation, Distribution, Yule
from aspartik.b3.substitutions import HKY
from aspartik.data.msa import MSA
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform


def make_mcmc(
    msa: MSA,
    *,
    trace_path: str,
    substitution_model: Literal["HKY"],
    operator_mix: Literal["default", "classic"] = "default",
    clock_rate: Optional[float] = None,
    tree_prior: Literal["yule", "constant"],
    seed: int = 4,
):
    rng = RNG(seed)

    parameters, operators, priors = [], [], []
    items = {}

    tree = Tree(msa.sequence_names(), rng)
    items["tree"] = tree
    parameters.append(tree)

    match substitution_model:
        case "HKY":
            kappa = Real(2.0)
            items["kappa"] = kappa
            frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
            items["frequencies"] = frequencies
            parameters.extend([kappa, frequencies])

            priors.extend(
                [
                    Distribution(kappa, LogNormal(1.0, 1.25)),
                ]
            )
            operators.extend(
                [
                    ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
                    DeltaExchange(frequencies, factor=0.01, rng=rng, weight=1),
                ]
            )

            sub_model = HKY(frequencies, kappa)

    match tree_prior:
        case "yule":
            birth_rate = Real(1.0)
            items["birth_rate"] = birth_rate
            parameters.append(birth_rate)
            operators.append(
                ParamScale(birth_rate, 0.75, Uniform(0, 1), rng, weight=3),
            )
            priors.extend(
                [
                    Distribution(birth_rate, LogNormal(1.0, 1.5)),
                    Yule(tree, birth_rate),
                ]
            )
        case "constant":
            population_size = Real(1.0)
            items["population_size"] = population_size
            parameters.append(population_size)
            operators.append(
                ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
            )
            priors.extend(
                [
                    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
                    ConstantPopulation(tree, population_size),
                ]
            )

    clock = None
    clock_rate_p = None
    match clock_rate:
        case None:
            clock_rate_p = Real(1.0)
            items["clock_rate"] = clock_rate_p
            operators.append(
                ParamScale(clock_rate_p, 0.75, Uniform(0, 1), rng, weight=3)
            )
            priors.append(Distribution(clock_rate_p, Laplace(0, 0.5)))
            parameters.append(clock_rate_p)
            clock = Clock.Strict(clock_rate_p)
        case float(clock_rate):
            clock_rate_p = Real(clock_rate)
            items["clock_rate"] = clock_rate_p
            parameters.append(clock_rate_p)
            clock = Clock.Strict(clock_rate_p)
    assert clock is not None
    assert clock_rate_p is not None

    match operator_mix:
        case "default":
            operators.extend(
                [
                    UpDown(
                        Internals(tree),
                        clock_rate_p,
                        0.75,
                        Uniform(0, 1),
                        rng,
                        weight=3,
                    ),
                    SubtreeLeap(tree, Normal(0, 1), rng, weight=msa.num_sequences),
                    FixedHeightSPR(tree, rng, weight=msa.num_sequences / 10),
                ]
            )
        case "classic":
            operators.extend(
                [
                    TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3),
                    SubtreeSlide(tree, Uniform(-0.5, 0.5), rng, weight=30),
                    BeastNarrowExchange(tree, rng, weight=30),
                    BeastWideExchange(tree, rng, weight=3),
                    RootSlide(tree, 0.75, Uniform(0, 1), rng, weight=3),
                    # BEAST's `UniformOperator` picks one of the parameter
                    # dimensions moves it randomly within bounds.  Using it on
                    # `nodeHeights` is equivalent to selecting a random node
                    # and moving it uniformly between it's maximum and minimum
                    # heights, which is what `NodeSlide` with `Uniform` does.
                    NodeSlide(tree, rng, weight=30),
                ]
            )

    likelihood = CPU4Likelihood(
        msa=msa,
        substitution=sub_model,
        clock=clock,
        tree=tree,
    )

    loggers = [
        PrintLogger(every=10_000),
        TraceWriter(items, trace_path, overwrite=True, zstd=True, every=1_000),
    ]

    mcmc = MCMC(
        state=parameters,
        priors=priors,
        operators=operators,
        likelihood=likelihood,
        callbacks=loggers,
        rng=rng,
    )

    return mcmc
