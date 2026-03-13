import pytest

from aspartik.b3.config import b3_config
from aspartik.io import read_msa_from_fasta

APES_MSA = read_msa_from_fasta("data/alignments/apes.fasta")


@pytest.mark.parametrize("calculator", ["cpu"])
@pytest.mark.parametrize("substitution_model", ["JC", "K80", "HKY"])  # TODO: GTR
@pytest.mark.parametrize("operator_mix", ["classic", "default"])
@pytest.mark.parametrize("clock_rate", [None, 1.0])
@pytest.mark.parametrize("tree_prior", ["yule", "constant", "exponential"])
@pytest.mark.parametrize("print_every", [None, 100])
@pytest.mark.parametrize("trace_path", [None, "target/test_config.trace"])
@pytest.mark.parametrize("timer", [False, True])
def test_config(
    calculator,
    substitution_model,
    operator_mix,
    clock_rate,
    tree_prior,
    print_every,
    trace_path,
    timer,
):
    mcmc = b3_config(
        APES_MSA,
        calculator=calculator,
        substitution_model=substitution_model,
        operator_mix=operator_mix,
        tree_prior=tree_prior,
        print_every=print_every,
        trace_path=trace_path,
        timer=timer,
    )

    mcmc.run(10)
