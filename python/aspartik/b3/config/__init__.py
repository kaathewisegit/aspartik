import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
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
    RandomWalk,
    RootSlide,
    ScaleReal,
    SubtreeLeap,
    SubtreeSlide,
    TreeScale,
    UpDown,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.priors import (
    Bound,
    ConstantPopulation,
    Distribution,
    ExponentialGrowth,
    SymmetricDirichlet,
    Yule,
)
from aspartik.b3.substitutions import GTR, HKY, JC, K80
from aspartik.b3.utils import print_operator_stats, print_operator_timings
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


@dataclass(slots=True)
class MCMCConfig:
    msa: MSA

    _: KW_ONLY

    heights: Optional[Sequence[float]] = None
    tree_sim_pop_size: float = 100.0

    tree_prior: Literal["yule", "constant", "exponential"]
    param_priors: dict[str, Continuous] = field(default_factory=dict)
    substitution_model: Literal["JC", "K80", "HKY", "GTR"]
    clock_rate: Optional[float] = None

    calculator: Literal["cpu", "cuda"] = "cpu"
    operator_mix: Literal["scalable", "beauti1_classic", "beauti1_default"] = "scalable"
    optimization_cutoff: int = 1_000_000

    overwrite: bool = True

    print_every: Optional[int] = 10_000

    trace_path: Optional[str] = None
    trace_every: int = 1_000

    # for BEAST
    trees_path: Optional[str] = None

    state_path: Optional[str] = None
    state_every: int = 100_000

    length: Optional[int] = None

    # printing
    print_stats: bool = False
    """b3-only setting to print acceptance/rejection stats per operator

    BEAST will always print stats regardless of this setting
    """
    print_timings: bool = False
    "b3-only setting to print the mean times of execution per operator"
    timer: bool = False

    seed: int = 4

    def b3_mcmc(self) -> MCMC:
        return _b3_config(self)

    def b3_run(self, mcmc: MCMC):
        assert self.length is not None
        mcmc.run(self.length)

        if self.print_stats or self.print_timings:
            print()
        if self.print_stats:
            print_operator_stats(mcmc)
        if self.print_stats and self.print_timings:
            print()
        if self.print_timings:
            print_operator_timings(mcmc)

    def b3_make_and_run(self):
        self.b3_run(self.b3_mcmc())

    def beast1_config(self) -> str:
        return _beast1_config(self)

    def beast1_run(self, config: str):
        assert self.length is not None
        _beast1_run(self, config)

    def beast1_make_and_run(self):
        self.beast1_run(self.beast1_config())


_beast1_default_template = """<?xml version="1.0" standalone="yes"?>

<beast version="10.5.0">
    <taxa id="taxa">
        {taxa}
    </taxa>

    <alignment id="alignment" dataType="nucleotide">
        {sequences}
    </alignment>

    <patterns id="patterns" from="1" strip="false">
        <alignment idref="alignment"/>
    </patterns>

    <constantSize id="_starting_coalescent" units="years">
        <populationSize>
            <parameter id="_starting_population_size" value="100.0" lower="0.0"/>
        </populationSize>
    </constantSize>
    <coalescentSimulator id="startingTree">
        <taxa idref="taxa"/>
        <constantSize idref="_starting_coalescent"/>
    </coalescentSimulator>

    <treeModel id="tree">
        <coalescentTree idref="startingTree"/>
        <rootHeight>
            <parameter id="tree.rootHeight"/>
        </rootHeight>
        <nodeHeights internalNodes="true">
            <parameter id="tree.internalNodeHeights"/>
        </nodeHeights>
        <nodeHeights internalNodes="true" rootNode="true">
            <parameter id="tree.allInternalNodeHeights"/>
        </nodeHeights>
    </treeModel>

{tree_prior}

{clock}

    <siteModel id="siteModel">
        <substitutionModel>
{substitution_model}
        </substitutionModel>
    </siteModel>

    <treeDataLikelihood id="treeLikelihood" useAmbiguities="false" usePreOrder="false">
        <partition>
            <patterns idref="patterns"/>
            <siteModel idref="siteModel"/>
        </partition>
        <treeModel idref="tree"/>
        <strictClockBranchRates idref="branchRates"/>
    </treeDataLikelihood>

    <operators id="operators" optimizationSchedule="log">
{operators}
    </operators>

    <mcmc id="mcmc" chainLength="{length}" autoOptimize="false" adaptation="false">
        <posterior id="posterior">
            <prior id="prior">
                {priors}

                <strictClockBranchRates idref="branchRates"/>
            </prior>
            <likelihood id="likelihood">
                <treeDataLikelihood idref="treeLikelihood"/>
            </likelihood>
        </posterior>
        <operators idref="operators"/>

        {screen_log}

        {file_log}

        {tree_log}
    </mcmc>

    <report>
        <property name="timer">
            <mcmc idref="mcmc"/>
        </property>
    </report>
</beast>
"""


def _beast1_config(c: MCMCConfig):
    operators, priors, log = "", "", ""

    if c.heights:
        taxa = [
            f'<taxon id="{name}">\n\t\t<date value="{height}" direction="backwards" units="years"/>\n\t</taxon>'
            for name, height in zip(c.msa.sequence_names(), c.heights)
        ]
    else:
        taxa = [f'<taxon id="{name}"/>' for name in c.msa.sequence_names()]
    taxa = "\n\t\t".join(taxa)

    sequences = []
    for i in range(c.msa.num_sequences):
        name = c.msa.sequence_name(i)
        seq = str(c.msa.sequence(i))
        sequences.append(
            f'<sequence>\n\t\t\t<taxon idref="{name}"/>\n\t\t\t{seq}\n\t\t</sequence>'
        )
    sequences = "\n\t\t".join(sequences)

    substitution_model_s = None
    match c.substitution_model:
        case "JC":
            # BEAST represents JC as a fixed HKY model
            substitution_model_s = """
    <HKYModel id="jc">
        <frequencies>
            <frequencyModel dataType="nucleotide">
                <frequencies>
                    <parameter id="frequencies" value="0.25 0.25 0.25 0.25"/>
                </frequencies>
            </frequencyModel>
        </frequencies>
        <kappa>
            <parameter id="kappa" value="1.0" />
        </kappa>
    </HKYModel>
"""

        case "HKY":
            substitution_model_s = """
    <HKYModel id="hky">
        <frequencies>
            <frequencyModel dataType="nucleotide">
                <frequencies>
                    <parameter id="frequencies" value="0.25 0.25 0.25 0.25"/>
                </frequencies>
            </frequencyModel>
        </frequencies>
        <kappa>
            <parameter id="kappa" value="2.0" lower="0.0"/>
        </kappa>
    </HKYModel>
"""
            operators += """
        <scaleOperator scaleFactor="0.75" weight="1">
            <parameter idref="kappa"/>
        </scaleOperator>
        <deltaExchange delta="0.01" weight="1">
            <parameter idref="frequencies"/>
        </deltaExchange>
            """

            priors += """
                <logNormalPrior id="prior.kappa" mu="1.0" sigma="1.25" offset="0.0">
                    <parameter idref="kappa"/>
                </logNormalPrior>
				<dirichletPrior alpha="1.0" sumsTo="1.0">
					<parameter idref="frequencies"/>
				</dirichletPrior>
            """

            log += """
            <parameter idref="kappa"/>
            <parameter idref="frequencies"/>
            """

        case "GTR":
            substitution_model_s = """
    <gtrModel id="gtr">
        <frequencies>
            <frequencyModel dataType="nucleotide">
                <frequencies>
                    <parameter id="frequencies" value="0.25 0.25 0.25 0.25"/>
                </frequencies>
            </frequencyModel>
        </frequencies>
        <rates>
            <parameter id="gtr.rates" dimension="6" value="1.0" lower="0.0"/>
        </rates>
    </gtrModel>
"""
            operators += """
		<deltaExchange delta="0.01" weight="1">
			<parameter idref="gtr.rates"/>
		</deltaExchange>
        <deltaExchange delta="0.01" weight="1">
            <parameter idref="frequencies"/>
        </deltaExchange>
            """

            priors += """
				<dirichletPrior alpha="1.0" sumsTo="6.0">
					<parameter idref="gtr.rates"/>
				</dirichletPrior>
				<dirichletPrior alpha="1.0" sumsTo="1.0">
					<parameter idref="frequencies"/>
				</dirichletPrior>
            """

            log += """
            <parameter idref="gtr.rates"/>
            <parameter idref="frequencies"/>
            """

    match c.operator_mix:
        case "scalable":
            num = min(c.msa.num_sequences, 1000)
            operators += f"""
		<scaleOperator scaleFactor="0.75" weight="3">
			<parameter idref="tree.rootHeight"/>
		</scaleOperator>
        <subtreeLeap size="1.0" weight="{num}">
            <treeModel idref="tree"/>
        </subtreeLeap>
        <fixedHeightSubtreePruneRegraft weight="{num / 10}">
            <treeModel idref="tree"/>
        </fixedHeightSubtreePruneRegraft>
            """
            pass
        case "beauti1_default":
            num = min(c.msa.num_sequences, 1000)
            operators += f"""
        <upDownOperator scaleFactor="0.75" weight="3">
            <up>
                <parameter idref="tree.allInternalNodeHeights"/>
            </up>
            <down>
                <parameter idref="clock_rate"/>
            </down>
        </upDownOperator>
        <subtreeLeap size="1.0" weight="{num}">
            <treeModel idref="tree"/>
        </subtreeLeap>
        <fixedHeightSubtreePruneRegraft weight="{num / 10}">
            <treeModel idref="tree"/>
        </fixedHeightSubtreePruneRegraft>
            """
        case "beauti1_classic":
            operators += """
		<scaleOperator scaleFactor="0.75" scaleAll="true" ignoreBounds="true" weight="3">
			<parameter idref="tree.allInternalNodeHeights"/>
		</scaleOperator>
		<subtreeSlide size="1.0" gaussian="true" weight="30">
			<treeModel idref="tree"/>
		</subtreeSlide>
		<narrowExchange weight="30">
			<treeModel idref="tree"/>
		</narrowExchange>
		<wideExchange weight="3">
			<treeModel idref="tree"/>
		</wideExchange>
		<wilsonBalding weight="3">
			<treeModel idref="tree"/>
		</wilsonBalding>
		<scaleOperator scaleFactor="0.75" weight="3">
			<parameter idref="tree.rootHeight"/>
		</scaleOperator>
		<uniformOperator weight="30">
			<parameter idref="tree.internalNodeHeights"/>
		</uniformOperator>
            """

    clock_s = None
    match c.clock_rate:
        case None:
            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="clock_rate"/>
        </scaleOperator>
            """

            priors += """
                <laplacePrior id="prior:clock_rate" mean="0.0" scale="0.5">
                    <parameter idref="clock_rate"/>
                </laplacePrior>
            """

            clock_s = """
    <strictClockBranchRates id="branchRates">
        <rate>
            <parameter id="clock_rate" value="1.0" lower="0.0"/>
        </rate>
    </strictClockBranchRates>
            """

            log += """
            <parameter idref="clock_rate"/>
            """

        case float(clock_rate):
            clock_s = f"""
    <strictClockBranchRates id="branchRates">
        <rate>
            <parameter id="clock_rate" value="{clock_rate}" lower="0.0"/>
        </rate>
    </strictClockBranchRates>
            """

    assert clock_s is not None

    tree_prior_s = None
    match c.tree_prior:
        case "constant":
            tree_prior_s = """
    <constantSize id="constant_population" units="years">
        <populationSize>
            <parameter id="population_size" value="1.0" lower="0.0"/>
        </populationSize>
    </constantSize>
    <coalescentLikelihood id="prior:coalescent">
        <model>
            <constantSize idref="constant_population"/>
        </model>
        <intervals>
            <treeIntervals>
                <treeModel idref="tree"/>
            </treeIntervals>
        </intervals>
    </coalescentLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="population_size"/>
        </scaleOperator>
            """

            priors += """
                <gammaPrior id="prior:population_size" shape="0.001" scale="1000.0" offset="0.0">
                    <parameter idref="population_size"/>
                </gammaPrior>

                <coalescentLikelihood idref="prior:coalescent"/>
            """

            log += """
            <parameter idref="population_size"/>
            <coalescentLikelihood idref="prior:coalescent"/>
            """

        case "exponential":
            tree_prior_s = """
    <exponentialGrowth id="exponential_growth" units="years">
        <populationSize>
            <parameter id="population_size" value="1.0" lower="0.0" />
        </populationSize>
        <growthRate>
            <parameter id="growth_rate" value="1.0" />
        </growthRate>
    </exponentialGrowth>
    <coalescentLikelihood id="prior:coalescent">
        <model>
            <exponentialGrowth idref="exponential_growth"/>
        </model>
        <intervals>
            <treeIntervals>
                <treeModel idref="tree"/>
            </treeIntervals>
        </intervals>
    </coalescentLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="population_size"/>
        </scaleOperator>
		<randomWalkOperator windowSize="1.0" weight="3">
			<parameter idref="growth_rate"/>
		</randomWalkOperator>
            """

            priors += """
                <gammaPrior id="prior:population_size" shape="0.001" scale="1000.0" offset="0.0">
                    <parameter idref="population_size"/>
                </gammaPrior>
				<laplacePrior mean="0" scale="100">
					<parameter idref="growth_rate"/>
				</laplacePrior>

                <coalescentLikelihood idref="prior:coalescent"/>
            """

            log += """
            <parameter idref="population_size"/>
            <parameter idref="growth_rate"/>
            <coalescentLikelihood idref="prior:coalescent"/>
            """

        case "yule":
            tree_prior_s = """
    <yulemodel id="yule" units="years">
        <birthRate>
            <parameter id="birth_rate" value="2.0" lower="0.0"/>
        </birthRate>
    </yulemodel>
    <speciationLikelihood id="prior:yule">
        <model>
            <yuleModel idref="yule"/>
        </model>
        <speciesTree>
            <treeModel idref="tree"/>
        </speciesTree>
    </speciationLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="birth_rate"/>
        </scaleOperator>
            """

            priors += """
                <logNormalPrior mu="1.0" sigma="1.5" offset="0.0">
                    <parameter idref="birth_rate"/>
                </logNormalPrior>
                <speciationLikelihood idref="prior:yule"/>
            """

            log += """
            <parameter idref="birth_rate"/>
            <speciationLikelihood idref="prior:yule"/>
            """

    assert tree_prior_s is not None

    file_log = (
        f"""
            <log id="fileLog" logEvery="{c.trace_every}" fileName="{c.trace_path}">
                <posterior idref="posterior"/>
                <prior idref="prior"/>
                <likelihood idref="likelihood"/>
                <treeHeightStatistic id="tree:height">
                    <treeModel idref="tree"/>
                </treeHeightStatistic>
                <treeLengthStatistic id="tree:length">
                    <treeModel idref="tree"/>
                </treeLengthStatistic>

    {log}
            </log>
    """
        if c.trace_path
        else ""
    )

    screen_log = (
        f"""
        <log id="screenLog" logEvery="{c.print_every}">
            <column label="Posterior" dp="4" width="12">
                <posterior idref="posterior"/>
            </column>
            <column label="Prior" dp="4" width="12">
                <prior idref="prior"/>
            </column>
            <column label="Likelihood" dp="4" width="12">
                <likelihood idref="likelihood"/>
            </column>
        </log>
    """
        if c.print_every
        else ""
    )

    tree_log = (
        f"""
        <logTree id="treeFileLog" logEvery="{c.trace_every}" fileName="{c.trees_path}">
            <treeModel idref="tree"/>
        </logTree>
    """
        if c.trees_path
        else ""
    )

    return _beast1_default_template.format(
        taxa=taxa,
        sequences=sequences,
        clock=clock_s,
        substitution_model=substitution_model_s,
        operators=operators,
        tree_prior=tree_prior_s,
        priors=priors,
        file_log=file_log,
        screen_log=screen_log,
        tree_log=tree_log,
        length=c.length,
    )


def _beast1_run(c: MCMCConfig, config: str):
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w+t") as tmp:
        tmp.write(config)
        tmp.flush()

        args = ["beast", "-seed", str(c.seed), "-citations_off"]
        if c.overwrite:
            args.append("-overwrite")
        match c.calculator:
            case "cpu":
                args.append("-beagle_CPU")
            case "cuda":
                args.append("-beagle_cuda")
        args.append(tmp.name)

        subprocess.run(args)


def _b3_config(c: MCMCConfig):
    rng = RNG(c.seed)

    parameters, operators, priors = [], [], []
    items = {}

    heights = c.heights
    if heights is None:
        heights = [0.0] * c.msa.num_sequences
    tree = Tree.simulate_coalescent(
        c.msa.sequence_names(), heights, c.tree_sim_pop_size, rng
    )
    items["tree"] = tree
    parameters.append(tree)

    def create_real(
        initial: float,
        name: str,
        dist: Continuous,
        weight: int,
        positive: bool = True,
    ):
        if name in c.param_priors:
            dist = c.param_priors[name]

        param = Real(initial)
        items[name] = param
        priors.append(Distribution(param, dist))
        parameters.append(param)
        if positive:
            operators.append(ScaleReal(param, Uniform(0, 1), rng, weight=weight))
        else:
            # a bigger window for the tuning parameter to do its work
            operators.append(RandomWalk(param, window=10, rng=rng, weight=weight))
        return param

    def create_frequencies():
        frequencies = RealVector(0.25, 0.25, 0.25, 0.25)
        items["frequencies"] = frequencies
        parameters.append(frequencies)
        operators.append(DeltaExchange(frequencies, rng=rng, weight=1))
        priors.append(SymmetricDirichlet(frequencies, 1))
        return frequencies

    match c.substitution_model:
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
            operators.append(DeltaExchange(rates, rng=rng, weight=1))
            priors.append(SymmetricDirichlet(rates, 6))

            sub_model = GTR(frequencies, rates)

    match c.tree_prior:
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
    match c.clock_rate:
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

    match c.operator_mix:
        case "scalable":
            num = min(c.msa.num_sequences, 1000)
            operators.extend(
                [
                    RootSlide(tree, Uniform(0, 1), rng, weight=3),
                    SubtreeLeap(tree, Normal(0, 1), rng, weight=num),
                    FixedHeightSPR(tree, rng, weight=num / 10),
                ]
            )
        case "beauti1_default":
            assert c.clock_rate is not None, (
                "BEAUti default operator mix doesn't support fixed clock rate"
            )

            num = min(c.msa.num_sequences, 1000)
            operators.extend(
                [
                    UpDown(
                        tree,
                        clock_rate_p,
                        Uniform(0, 1),
                        rng,
                        weight=3,
                    ),
                    SubtreeLeap(tree, Normal(0, 1), rng, weight=num),
                    FixedHeightSPR(tree, rng, weight=num / 10),
                ]
            )
        case "beauti1_classic":
            operators.extend(
                [
                    TreeScale(tree, Uniform(0, 1), rng, weight=3),
                    SubtreeSlide(tree, Uniform(-0.5, 0.5), rng, weight=30),
                    BeastNarrowExchange(tree, rng, weight=30),
                    BeastWideExchange(tree, rng, weight=3),
                    RootSlide(tree, Uniform(0, 1), rng, weight=3),
                    NodeSlide(tree, rng, weight=30),
                ]
            )

    match c.calculator:
        case "cpu":
            likelihood = CPU4Likelihood(
                msa=c.msa, substitution=sub_model, clock=clock, tree=tree
            )
        case "cuda":
            likelihood = CUDALikelihood(
                msa=c.msa, substitution=sub_model, clock=clock, tree=tree
            )

    callbacks = []
    if c.print_every:
        callbacks.append(PrintLogger(every=c.print_every))
    if c.trace_path:
        callbacks.append(
            TraceWriter(
                items,
                c.trace_path,
                overwrite=c.overwrite,
                zstd=True,
                every=c.trace_every,
            )
        )
    if c.state_path:
        callbacks.append(StateCheckpoint(c.state_path, every=c.state_every))
    if c.timer:
        callbacks.append(Timer())

    mcmc = MCMC(
        priors=priors,
        operators=operators,
        likelihood=likelihood,
        callbacks=callbacks,
        rng=rng,
        optimization_cutoff=c.optimization_cutoff,
    )

    return mcmc
