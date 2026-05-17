import pytest

from aspartik.b3 import MCMC, Clock
from aspartik.b3.config import MCMCConfig
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.operators import ParamScale, TreeScale
from aspartik.b3.parameters import Real, Tree
from aspartik.b3.priors import Bound
from aspartik.b3.substitutions import K80
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Uniform


def test_mcmc():
    msa = read_msa_from_fasta("data/alignments/apes.fasta")

    rng = RNG(4)
    tree = Tree(msa.sequence_names(), rng)

    a = Real(2.0)

    prior_a = Bound(a, 0, 10)

    op_param_scale = ParamScale(a, Uniform(0, 1), rng, weight=1)
    op_tree_scale = TreeScale(tree, Uniform(0, 1), rng, weight=3)

    likelihood = CPU4Likelihood(
        msa=msa,
        substitution=K80(Real(1.0)),
        clock=Clock.Strict(Real(1.0)),
        tree=tree,
    )

    mcmc = MCMC(
        priors=[prior_a],
        operators=[op_param_scale, op_tree_scale],
        likelihood=likelihood,
        callbacks=[],
        rng=rng,
    )

    mcmc.run(4)

    assert mcmc.priors == [prior_a]
    assert mcmc.operators == [op_param_scale, op_tree_scale]
    assert mcmc.likelihood == likelihood
    assert mcmc.rng == rng

    assert mcmc.current_step == 4


def get_mcmc():
    msa = read_msa_from_fasta("data/alignments/apes.fasta")

    return MCMCConfig(
        msa,
        calculator="cpu",
        substitution_model="GTR",
        tree_prior="constant",
    ).b3_mcmc()


def test_dump_restore():
    mcmc0 = get_mcmc()
    mcmc0.run(10_000)
    state = mcmc0.dump_state()
    mcmc0.run(10_000)

    mcmc1 = get_mcmc()
    mcmc1.load_state(state)
    mcmc1.run(10_000)

    tree0, tree1 = mcmc0.parameters[4], mcmc1.parameters[4]
    assert isinstance(tree0, Tree)
    assert isinstance(tree1, Tree)
    assert tree0.to_newick() == tree1.to_newick()


def test_version_assertion():
    mcmc = get_mcmc()
    mcmc.run(10_000)
    state = mcmc.dump_state()

    state = state[:4] + b"9" + state[5:]

    with pytest.raises(match="Tried to load state made by Aspartik version 9"):
        mcmc.load_state(state)
