from aspartik.b3 import MCMC, Likelihood, Tree
from aspartik.b3.clocks import StrictClock
from aspartik.b3.operators import ParamScale, TreeScale
from aspartik.b3.parameters import Integer, Real
from aspartik.b3.priors import Bound
from aspartik.b3.substitutions import K80
from aspartik.io.msa import read_msa_from_fasta
from aspartik.rng import RNG
from aspartik.stats.distributions import Uniform


def test_mcmc():
    msa = read_msa_from_fasta("crates/b3/data/apes.fasta")

    rng = RNG(4)
    tree = Tree(msa.sequence_names(), rng)

    a = Real(2.0)

    prior_a = Bound(a, 0, 10)

    op_param_scale = ParamScale(a, 0.75, Uniform(0, 1), rng, weight=1)
    op_tree_scale = TreeScale(tree, 0.75, Uniform(0, 1), rng, weight=3)

    likelihood = Likelihood(
        msa=msa,
        substitution=K80(1.0),
        clock=StrictClock(1.0),
        tree=tree,
        calculator="cpu",
    )

    mcmc = MCMC(
        burnin=0,
        length=4,
        state=[a, tree],
        priors=[prior_a],
        operators=[op_param_scale, op_tree_scale],
        likelihoods=[likelihood],
        callbacks=[],
        rng=rng,
    )

    mcmc.run()

    assert mcmc.priors == [prior_a]
    assert mcmc.operators == [op_param_scale, op_tree_scale]
    assert mcmc.likelihoods == [likelihood]
    assert mcmc.rng == rng

    assert mcmc.current_step == 4
