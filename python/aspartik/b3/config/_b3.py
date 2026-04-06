from collections.abc import Sequence
from typing import Literal, Optional

from aspartik.b3 import MCMC, Clock
from aspartik.b3.callbacks import PrintLogger, StateCheckpoint, Timer, TraceWriter
from aspartik.b3.likelihoods import CPU4Likelihood, CUDALikelihood
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
    SymmetricDirichlet,
    Yule,
)
from aspartik.b3.substitutions import GTR, HKY, JC, K80
from aspartik.data.msa import MSA
from aspartik.rng import RNG
from aspartik.stats.distributions import (
    Continuous,
    Gamma,
    Laplace,
    LogNormal,
    Normal,
    Uniform,
)

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
    param_priors: dict[str, Continuous] = {},
    print_every: Optional[int] = 1_000,
    trace_path: Optional[str] = None,
    trace_every: int = 1_000,
    state_path: Optional[str] = None,
    state_every: int = 100_000,
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

    def create_real(
        initial: float,
        name: str,
        dist: Continuous,
        weight: int,
        positive: bool = True,
    ):
        if name in param_priors:
            dist = param_priors[name]

        param = Real(initial)
        items[name] = param
        priors.append(Distribution(param, dist))
        parameters.append(param)
        if positive:
            operators.append(ParamScale(param, 0.75, Uniform(0, 1), rng, weight=weight))
        else:
            operators.append(RandomWalk(param, window=1, rng=rng, weight=weight))
        return param

    def create_frequencies():
        frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
        items["frequencies"] = frequencies
        parameters.append(frequencies)
        operators.append(DeltaExchange(frequencies, factor=0.01, rng=rng, weight=1))
        priors.append(SymmetricDirichlet(frequencies, 1))
        return frequencies

    match substitution_model:
        case "JC":
            sub_model = JC()
        case "K80":
            kappa = create_real(2.0, "kappa", LogNormal(1.0, 1.25), weight=1)
            sub_model = K80(kappa)
        case "HKY":
            kappa = create_real(2.0, "kappa", LogNormal(1.0, 1.25), weight=1)
            frequencies = create_frequencies()

            sub_model = HKY(frequencies, kappa)
        case "GTR":
            frequencies = create_frequencies()
            rates = RealVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            items["rates"] = rates
            parameters.append(rates)
            operators.append(DeltaExchange(rates, factor=0.01, rng=rng, weight=1))
            priors.append(SymmetricDirichlet(rates, 6))

            sub_model = GTR(frequencies, rates)

    match tree_prior:
        case "yule":
            birth_rate = create_real(1.0, "birth_rate", LogNormal(1.0, 1.5), weight=3)
            yule = Yule(tree, birth_rate)
            items["prior:yule"] = yule
            priors.append(yule)
        case "constant":
            population_size = create_real(
                1.0, "population_size", Gamma(0.001, 1 / 1000.0), weight=3
            )
            coalescent = ConstantPopulation(tree, population_size)
            items["prior:coalescent"] = coalescent
            priors.append(coalescent)
        case "exponential":
            population_size = create_real(
                1.0, "population_size", Gamma(0.001, 1 / 1000.0), weight=3
            )
            growth_rate = create_real(
                1.0, "growth_rate", Laplace(0, 100), weight=3, positive=False
            )

            coalescent = ExponentialGrowth(tree, population_size, growth_rate)
            items["prior:coalescent"] = coalescent
            priors.extend([Bound(growth_rate), coalescent])

    clock = None
    clock_rate_p = None
    match clock_rate:
        case None:
            clock_rate_p = create_real(1.0, "clock_rate", Laplace(0, 0.5), weight=3)
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
            if clock_rate is None:
                operators.append(
                    UpDown(
                        Internals(tree),
                        clock_rate_p,
                        0.75,
                        Uniform(0, 1),
                        rng,
                        weight=3,
                    )
                )
            else:
                operators.append(TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3))
            operators.extend(
                [
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
    if state_path:
        callbacks.append(StateCheckpoint(state_path, every=state_every))
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
