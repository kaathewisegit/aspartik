from aspartik.b3 import MCMC, Clock
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.operators import ParamScale, TreeScale
from aspartik.b3.parameters import Real, Tree
from aspartik.b3.priors import Bound
from aspartik.b3.substitutions import K80
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Uniform


def test_mcmc():
    msa = read_msa_from_fasta("data/alignments/apes.fasta")

    rng = RNG(4)
    tree = Tree(msa.sequence_names(), rng)

    a = Real(2.0)

    prior_a = Bound(a, 0, 10)

    op_param_scale = ParamScale(a, 0.75, Uniform(0, 1), rng, weight=1)
    op_tree_scale = TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3)

    likelihood = CPU4Likelihood(
        msa=msa,
        substitution=K80(Real(1.0)),
        clock=Clock.Strict(1.0),
        tree=tree,
    )

    mcmc = MCMC(
        state=[a, tree],
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
