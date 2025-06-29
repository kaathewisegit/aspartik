from mpmath import mp
import numpy as np

from aspartik.math import is_close
from aspartik.math.functions import erf, erfc, erf_inv, erfc_inv

mp.dps = 1000


def test_erf():
    for x in np.arange(0.0, 5.0, 0.1, dtype=float):
        assert is_close(erf(x), float(mp.erf(x)), relative=1e-10)


def test_erfc():
    for x in np.arange(0.0, 5.0, 0.1, dtype=float):
        assert is_close(erfc(x), float(mp.erfc(x)), relative=1e-9)


def test_erf_inv():
    for x in np.arange(0.0, 1.0, 0.05, dtype=float):
        assert is_close(erf_inv(x), float(mp.erfinv(x)))


def test_erfc_inv():
    for x in np.arange(0.0, 1.0, 0.05, dtype=float):
        expected = float(mp.erfinv(mp.mpf("1.0") - x))
        assert is_close(erfc_inv(x), expected)
