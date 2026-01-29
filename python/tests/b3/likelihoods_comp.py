import pytest

from dataclasses import dataclass

from aspartik.b3 import MCMC, Callback, Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    HeteroLikelihood,
    Likelihood,
    Parallel4Likelihood,
)
from aspartik.b3.loggers import PrintLogger
from aspartik.b3.operators import (
    DeltaExchange,
    FixedHeightSubtreePruneRegraft,
    ParamScale,
    RandomWalk,
    SubtreeLeap,
    UpDown,
)
from aspartik.b3.parameters import Internals, Real, RealVector, Tree
from aspartik.b3.priors import Bound, Distribution, ExponentialGrowth, Yule
from aspartik.b3.substitutions import HKY
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Gamma, Laplace, LogNormal, Uniform


@pytest.mark.manual
def test_compare_likelihood():
    rng = RNG(4)

    msa = read_msa_from_fasta("data/alignments/influenza.fasta")

    tree = Tree(msa.sequence_names(), rng)

    kappa = Real(2.0)
    population_size = Real(1.0)
    growth_rate = Real(0)
    clock_rate = Real(0.001)
    frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
    params = [kappa, population_size, growth_rate, clock_rate, frequencies]

    priors = [
        Bound(kappa),
        Bound(population_size),
        Bound(clock_rate),
        Bound(frequencies),
        Distribution(kappa, LogNormal(1.0, 1.25)),
        Distribution(clock_rate, Laplace(0, 0.5)),
        Distribution(population_size, Gamma(0.001, 1 / 1000.0)),
        Distribution(growth_rate, Laplace(0, 100)),
        ExponentialGrowth(tree, population_size, growth_rate),
    ]

    operators = [
        ParamScale(kappa, 0.75, Uniform(0, 1), rng, weight=1),
        ParamScale(clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
        UpDown(Internals(tree), clock_rate, 0.75, Uniform(0, 1), rng, weight=3),
        SubtreeLeap(tree, Uniform(0, 1), rng, weight=msa.num_sequences),
        FixedHeightSubtreePruneRegraft(tree, rng, weight=msa.num_sequences / 10),
        ParamScale(population_size, 0.75, Uniform(0, 1), rng, weight=3),
        RandomWalk(growth_rate, window=1, rng=rng, weight=3),
        DeltaExchange(frequencies, 0.01, rng, weight=3),
    ]

    cpu_calculator = CPU4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
    )
    parallel_calculator = Parallel4Likelihood(
        msa=msa,
        substitution=HKY(frequencies, kappa),
        clock=Clock.Strict(clock_rate),
        tree=tree,
        num_leaf_threads=5,
        num_internal_threads=2,
    )
    hetero = HeteroLikelihood(
        likelihoods=[
            CPU4Likelihood(
                msa=msa,
                substitution=HKY(frequencies, kappa),
                clock=Clock.Strict(clock_rate),
                tree=tree,
            )
        ]
    )
    try:
        cuda_calculator = CUDALikelihood(
            msa=msa,
            substitution=HKY(frequencies, kappa),
            clock=Clock.Strict(clock_rate),
            tree=tree,
        )
    except Exception as e:
        cuda_calculator = None

    calculators = [cpu_calculator, parallel_calculator, hetero]
    if cuda_calculator:
        calculators.insert(0, cuda_calculator)

    @dataclass
    class TestLikelihood(Likelihood):
        calcs: list[Likelihood]

        def propose(self):
            for calc in self.calcs:
                calc.propose()

        def likelihood(self):
            for calc in self.calcs:
                calc.likelihood()

            return self.calcs[1].likelihood()

        def accept(self):
            for calc in self.calcs:
                calc.accept()

        def reject(self):
            for calc in self.calcs:
                calc.reject()

    test_likelihood = TestLikelihood(calculators)

    class CheckLikelihood(Callback):
        every: int = 1

        max_diff: float = 0.0

        def call(self, mcmc: MCMC):
            likelihoods = [calculator.likelihood() for calculator in calculators]
            diff = max(likelihoods) - min(likelihoods)
            self.max_diff = max(self.max_diff, diff)

    mcmc = MCMC(
        state=params + [tree],
        priors=priors,
        operators=operators,
        likelihood=test_likelihood,
        callbacks=[CheckLikelihood()],
        rng=rng,
    )

    mcmc.run(100_000)
    likelihood = mcmc.likelihood.likelihood()
    assert mcmc.callbacks[0].max_diff < abs(likelihood * 0.01)
