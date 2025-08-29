import pytest
from mpmath import mp
from utils import random_float

from aspartik.math import is_close
from aspartik.math.functions import gamma, ln_gamma

mp.dps = 1000


@pytest.mark.parametrize("x", random_float(0, 20, num=1000))
def test_gamma(x):
    assert is_close(gamma(x), float(mp.gamma(x)), relative=1e-12)


@pytest.mark.parametrize("x", random_float(0, 1000, num=1000))
def test_ln_gamma(x):
    assert is_close(ln_gamma(x), float(mp.loggamma(x)), relative=1e-11)


# These test cases were missed by `num=100`, but got caught with `num=1000`
@pytest.mark.parametrize(
    "x",
    [
        0.997705202483382,
        0.9997328880840233,
        0.9986221322650661,
    ],
)
def test_ln_gamma_close_to_1(x):
    assert is_close(ln_gamma(x), float(mp.loggamma(x)), relative=1e-11)
