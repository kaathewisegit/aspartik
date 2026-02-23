import pytest
from utils import random_float, random_integer

from aspartik.rng import RNG


@pytest.fixture
def rng():
    return RNG(4)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.0, 1.0),
        (0.0, 10.0),
        (1e10, 1e20),
        (-1.0, 0.0),
        (-1e10, 1e10),
        (-1.0, 1.0),
        (1.0, 2.0),
    ],
)
def test_float_in_bounds(rng, upper, lower):
    for _ in range(1000):
        assert lower <= rng.random_float(lower=lower, upper=upper) <= upper


@pytest.mark.parametrize("seed", random_integer(0, 2**63 - 1))
def test_seed(seed: int):
    rng = RNG(seed)
    rng.random_bool()


@pytest.mark.parametrize("ratio", [0, 1, *random_float(0, 1)])
def test_bool(rng, ratio: float):
    rng.random_bool(ratio)


@pytest.mark.parametrize(
    "ratio", [*random_float(1, 10**10), *random_float(-(10**10), 0)]
)
def test_bool_error(rng, ratio: float):
    with pytest.raises(Exception) as e:
        rng.random_bool(ratio)
