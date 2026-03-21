import mpmath
import pytest
from utils import random_integer

from aspartik.math import is_close
from aspartik.stats.distributions import Poisson

mpmath.mp.prec = 1000


def test_basic():
    d = Poisson(1)
    assert d.lambda_ == 1
    assert repr(d) == "Poisson(1)"
    assert is_close(0.367879441171442, d.pmf(1), relative=1e-15)
    assert d.lower == 0
    assert d.upper == 2**64 - 1  # u64::MAX


@pytest.mark.parametrize("lambda_", [1, 4, 10])
@pytest.mark.parametrize("x", random_integer(0, 20, num=250))
def test_pmf(lambda_, x):
    d = Poisson(lambda_)

    mp_lambda = mpmath.mpf(lambda_)
    mp_x = mpmath.mpf(x)
    expected = (mp_lambda**mp_x * mpmath.e ** (-mp_lambda)) / mpmath.factorial(mp_x)
    expected = float(expected)

    assert is_close(d.pmf(x), expected, relative=1e-14)


@pytest.mark.parametrize("lambda_", [1, 4, 10])
@pytest.mark.parametrize("x", random_integer(0, 20, num=250))
def test_cdf(lambda_, x):
    d = Poisson(lambda_)

    mp_lambda = mpmath.mpf(lambda_)
    mp_x = mpmath.mpf(x)
    expected = mpmath.gammainc(mp_x + 1, mp_lambda, regularized=True)
    expected = float(expected)

    assert is_close(d.cdf(x), expected, relative=1e-13)
