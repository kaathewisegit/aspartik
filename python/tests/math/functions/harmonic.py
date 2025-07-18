from mpmath import mp
import pytest

from aspartik.math import is_close
from aspartik.math.functions import harmonic

mp.dps = 1000


@pytest.mark.parametrize("x", range(1, 100))
def test_harmonic(x):
    assert is_close(harmonic(x), float(mp.harmonic(x)), relative=1e-14)


def test_harmonic_special():
    assert harmonic(0) == 1
