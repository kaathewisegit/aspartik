import pytest

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
