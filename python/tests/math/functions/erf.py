import mpmath
import pytest
from utils import random_float

from aspartik.math import is_close
from aspartik.math.functions import erf, erf_inv, erfc, erfc_inv

mpmath.mp.prec = 1000


@pytest.mark.parametrize("x", random_float(0, 5))
def test_erf(x):
    assert is_close(erf(x), float(mpmath.erf(x)), relative=1e-10)


@pytest.mark.parametrize("x", random_float(0, 5))
def test_erfc(x):
    assert is_close(erfc(x), float(mpmath.erfc(x)), relative=1e-9)


@pytest.mark.parametrize("x", random_float(0, 1))
def test_erf_inv(x):
    assert is_close(erf_inv(x), float(mpmath.erfinv(x)))


@pytest.mark.parametrize("x", random_float(0, 1))
def test_erfc_inv(x):
    expected = float(mpmath.erfinv(mpmath.mpf("1.0") - x))
    assert is_close(erfc_inv(x), expected)
