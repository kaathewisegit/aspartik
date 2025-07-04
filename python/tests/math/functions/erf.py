from mpmath import mp
import pytest
from tests.utils import random_float

from aspartik.math import is_close
from aspartik.math.functions import erf, erfc, erf_inv, erfc_inv

mp.dps = 1000


@pytest.mark.parametrize("x", random_float(0, 5))
def test_erf(x):
    assert is_close(erf(x), float(mp.erf(x)), relative=1e-10)


@pytest.mark.parametrize("x", random_float(0, 5))
def test_erfc(x):
    assert is_close(erfc(x), float(mp.erfc(x)), relative=1e-9)


@pytest.mark.parametrize("x", random_float(0, 1))
def test_erf_inv(x):
    assert is_close(erf_inv(x), float(mp.erfinv(x)))


@pytest.mark.parametrize("x", random_float(0, 1))
def test_erfc_inv(x):
    expected = float(mp.erfinv(mp.mpf("1.0") - x))
    assert is_close(erfc_inv(x), expected)
