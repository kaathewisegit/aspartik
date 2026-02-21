import pytest
from utils.compare import compare_verify_run

from aspartik.b3.parameters import Real, Tree
from aspartik.b3.priors import ExponentialGrowth
from aspartik.rng import RNG


@pytest.mark.skip
def test_exponential_growth():
    tree = Tree(["a", "b"], RNG(0))
    population_size = Real(0)
    growth_rate = Real(0)
    eg = ExponentialGrowth(tree, population_size, growth_rate)

    compare_verify_run(
        "data/runs/respiratory/log",
        # "data/runs/respiratory/trees",
        parameters={
            "tree": tree,
            "population_size": population_size,
            "growth_rate": growth_rate,
        },
        priors={
            "prior:coalescent": eg,
        },
    )
