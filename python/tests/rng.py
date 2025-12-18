import pytest
from utils import random_integer

import pickle

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


CLONG_MAX = 2**32 - 1


@pytest.mark.parametrize("seed", random_integer(0, CLONG_MAX, num=1000))
def test_pickle(seed: int):
    rng = RNG(seed)
    for _ in range(100):
        _ = rng.random_int(0, CLONG_MAX)
        _ = rng.random_float()
        _ = rng.random_bool()

    pickled = pickle.dumps(rng)
    copy = pickle.loads(pickled)

    for _ in range(100):
        a, b = rng.random_int(0, CLONG_MAX), copy.random_int(0, CLONG_MAX)
        assert a == b

        a, b = rng.random_float(), copy.random_float()
        assert a == b

        a, b = rng.random_bool(), copy.random_bool()
        assert a == b
