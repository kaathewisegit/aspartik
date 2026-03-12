from collections.abc import Sequence
from typing import Literal, Optional

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import Timer, TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood, CUDALikelihood
from aspartik.b3.loggers import PrintLogger
from aspartik.b3.operators import (
    BeastNarrowExchange,
    BeastWideExchange,
    DeltaExchange,
    FixedHeightSPR,
    NodeSlide,
    ParamScale,
    RandomWalk,
    RootSlide,
    SubtreeLeap,
    SubtreeSlide,
    TreeScale,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import (
    Bound,
    ConstantPopulation,
    Distribution,
    ExponentialGrowth,
    Yule,
)
from aspartik.b3.substitutions import HKY, JC
from aspartik.data.msa import MSA
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Normal, Uniform

from ._shared import CalculatorKind, SubstitutionModel, TreePrior


def b3_config(
    msa: MSA,
    *,
    heights: Optional[Sequence] = None,
    calculator: CalculatorKind = "cpu",
    substitution_model: SubstitutionModel,
    operator_mix: Literal["default", "classic"] = "default",
    clock_rate: Optional[float] = None,
    tree_prior: TreePrior,
    print_every: Optional[int] = 1_000,
    trace_path: Optional[str] = None,
    trace_every: int = 1_000,
    timer: bool = False,
    seed: int = 4,
):
    rng = RNG(seed)

    parameters, operators, priors = [], [], []
    items = {}

    tree = Tree(msa.sequence_names(), rng)
    items["tree"] = tree
    parameters.append(tree)

    if heights:
        for leaf, height in zip(tree.leaves(), heights):
            tree.set_height(leaf, height)
        tree.set_random_heights(0.3, rng)
        tree.accept()

    match substitution_model:
        case "JC":
            sub_model = JC()
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
            yule = Yule(tree, birth_rate)
            items["prior:yule"] = yule
            priors.extend([Distribution(birth_rate, LogNormal(1.0, 1.5)), yule])
        case "constant":
            population_size = Real(1.0)
            items["population_size"] = population_size
            parameters.append(population_size)
            operators.append(
                ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
            )
            coalescent = ConstantPopulation(tree, population_size)
            items["prior:coalescent"] = coalescent
            priors.extend(
                [Distribution(population_size, Gamma(0.001, 1 / 1000.0)), coalescent]
            )
        case "exponential":
            population_size = Real(1.0)
            items["population_size"] = population_size
            growth_rate = Real(1.0)
            items["growth_rate"] = growth_rate
            parameters.extend([population_size, growth_rate])

            operators.extend(
                [
                    ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
                    RandomWalk(growth_rate, window=1, rng=rng, weight=3),
                ]
            )
            priors.extend(
                [
                    Bound(growth_rate),
                    Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
                    Distribution(growth_rate, Laplace(0, 100)),
                    ExponentialGrowth(tree, population_size, growth_rate),
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
            num = min(msa.num_sequences, 1000)
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
                    SubtreeLeap(tree, Normal(0, 1), rng, weight=num),
                    FixedHeightSPR(tree, rng, weight=num / 10),
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

    match calculator:
        case "cpu":
            likelihood = CPU4Likelihood(
                msa=msa, substitution=sub_model, clock=clock, tree=tree
            )
        case "cuda":
            likelihood = CUDALikelihood(
                msa=msa, substitution=sub_model, clock=clock, tree=tree
            )

    callbacks = []
    if print_every:
        callbacks.append(PrintLogger(every=print_every))
    if trace_path:
        callbacks.append(
            TraceWriter(items, trace_path, overwrite=True, zstd=True, every=trace_every)
        )

    if timer:
        callbacks.append(Timer())

    mcmc = MCMC(
        state=parameters,
        priors=priors,
        operators=operators,
        likelihood=likelihood,
        callbacks=callbacks,
        rng=rng,
    )

    return mcmc
