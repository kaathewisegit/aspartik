import pytest
from mpmath import findroot, gammainc, mp
from utils import random_float

from math import inf

from aspartik.math import is_close
from aspartik.stats.distributions import Gamma, GammaError

mp.dps = 200


def gamma_cdf(x, shape, scale=1.0):
    return gammainc(shape, 0, x / scale, regularized=True)


def gamma_inverse_cdf(p, shape, scale=1.0):
    return findroot(
        lambda x: gamma_cdf(x, shape, scale) - p, shape, solver="halley", tol=1e-15
    )


def test_basic():
    g = Gamma(1, 2)
    assert g.shape == 1
    assert g.rate == 2
    assert repr(g) == "Gamma(shape=1, rate=2)"
    assert g.pdf(0.5) == 0.7357588823428847
    assert g.lower == 0
    assert g.upper == inf


def test_errors():
    with pytest.raises(ValueError) as error:
        Gamma(-2, 1)
    assert error.value.args[0] == GammaError.ShapeInvalid

    with pytest.raises(ValueError) as error:
        Gamma(1, -2)
    assert error.value.args[0] == GammaError.RateInvalid


@pytest.mark.parametrize("shape, rate", [(1, 2), (2.5, 1.5), (0.5, 3.0), (10, 0.1)])
@pytest.mark.parametrize("x", random_float(1e-5, 20, num=100))
def test_gamma_cdf(shape, rate, x):
    g = Gamma(shape, rate)
    scale = 1 / rate
    expected = gamma_cdf(mp.mpf(str(x)), shape, scale)
    assert is_close(g.cdf(x), float(expected), relative=1e-12)


@pytest.mark.parametrize("shape, rate", [(1, 2), (2.5, 1.5)])
@pytest.mark.parametrize("x", random_float(0.1, 1, num=50))
def test_gamma_inverse_cdf(shape, rate, x):
    g = Gamma(shape, rate)
    scale = 1 / rate
    expected = gamma_inverse_cdf(mp.mpf(str(x)), shape, scale)
    assert is_close(g.inverse_cdf(x), float(expected), relative=1e-12)
