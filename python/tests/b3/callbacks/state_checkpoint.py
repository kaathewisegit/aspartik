import pytest
from utils import random_msas

from aspartik.b3 import MCMC
from aspartik.b3.config import MCMCConfig
from aspartik.data.msa import MSA


@pytest.mark.parametrize("msa", random_msas(10, 100, num=2))
def test_state_restore(msa: MSA, tmp_path):
    state_path = tmp_path / "test.state"

    def make_mcmc() -> MCMC:
        return MCMCConfig(
            msa,
            calculator="cpu",
            tree_prior="constant",
            substitution_model="HKY",
            state_path=state_path,
            state_every=100,
            print_every=None,
        ).b3_mcmc()

    mcmc = make_mcmc()
    mcmc.run(10_000)
    assert mcmc.current_step == 10_000

    with open(state_path, "rb") as file:
        state = file.read()

    mcmc = make_mcmc()
    mcmc.run(2_000)
    assert mcmc.current_step == 2_000

    mcmc.load_state(state)
    mcmc.run(3_000)
    assert mcmc.current_step == 13_000
