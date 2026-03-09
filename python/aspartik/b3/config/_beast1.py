import subprocess
import tempfile
from collections.abc import Sequence
from typing import Literal, Optional

from aspartik.data.msa import MSA

from ._shared import CalculatorKind, SubstitutionModel, TreePrior

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

{substitution_model}

    <siteModel id="siteModel">
        <substitutionModel>
            <HKYModel idref="hky"/>
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
    </mcmc>

    <report>
        <property name="timer">
            <mcmc idref="mcmc"/>
        </property>
    </report>
</beast>
"""


def _file_log(log: str, log_path: Optional[str]) -> str:
    if not log_path:
        return ""

    return f"""
            <log id="fileLog" logEvery="1000" fileName="{log_path}">
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


def _screen_log(screen_log_every: Optional[int]) -> str:
    if not screen_log_every:
        return ""

    return f"""
        <log id="screenLog" logEvery="{screen_log_every}">
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


def beast1_config(
    msa: MSA,
    *,
    heights: Optional[Sequence] = None,
    substitution_model: SubstitutionModel,
    operator_mix: Literal["default", "classic"] = "default",
    clock_rate: Optional[float] = None,
    tree_prior: TreePrior,
    log_path: Optional[str] = None,
    screen_log_every: Optional[int] = 1_000,
    length: int,
):
    operators, priors, log = "", "", ""

    if heights:
        taxa = [
            f'<taxon id="{name}">\n\t\t<date value="{height}" direction="backwards" units="years"/>\n\t</taxon>'
            for name, height in zip(msa.sequence_names(), heights)
        ]
    else:
        taxa = [f'<taxon id="{name}"/>' for name in msa.sequence_names()]
    taxa = "\n\t\t".join(taxa)

    sequences = []
    for i in range(msa.num_sequences):
        name = msa.sequence_name(i)
        seq = str(msa.sequence(i))
        sequences.append(
            f'<sequence>\n\t\t\t<taxon idref="{name}"/>\n\t\t\t{seq}\n\t\t</sequence>'
        )
    sequences = "\n\t\t".join(sequences)

    substitution_model_s = None
    match substitution_model:
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
        <deltaExchange delta="0.01" weight="3">
            <parameter idref="frequencies"/>
        </deltaExchange>
            """

            priors += """
                <logNormalPrior id="prior.kappa" mu="1.0" sigma="1.25" offset="0.0">
                    <parameter idref="kappa"/>
                </logNormalPrior>
            """

            log += """
            <parameter idref="kappa"/>
            <parameter idref="frequencies"/>
            """

    match operator_mix:
        case "default":
            num = min(msa.num_sequences, 1000)
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

    clock_s = None
    match clock_rate:
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
    match tree_prior:
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

    return _beast1_default_template.format(
        taxa=taxa,
        sequences=sequences,
        clock=clock_s,
        substitution_model=substitution_model_s,
        operators=operators,
        tree_prior=tree_prior_s,
        priors=priors,
        file_log=_file_log(log, log_path),
        screen_log=_screen_log(screen_log_every),
        length=length,
    )


def beast1_run(
    config: str,
    calculator: CalculatorKind = "cpu",
    overwrite: bool = True,
    seed: int = 4,
):
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w+t") as tmp:
        tmp.write(config)
        tmp.flush()

        args = ["beast", "-seed", str(seed), "-citations_off"]
        if overwrite:
            args.append("-overwrite")
        match calculator:
            case "cpu":
                args.append("-beagle_CPU")
                args.append("-beagle_threading_off")
            case "parallel":
                args.append("-beagle_CPU")
            case "cuda":
                args.append("-beagle_cuda")
        args.append(tmp.name)

        subprocess.run(args)
