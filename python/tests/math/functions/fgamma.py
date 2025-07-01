from mpmath import mp
import numpy as np

from aspartik.math import is_close
from aspartik.math.functions import gamma, ln_gamma

mp.dps = 1000


def test_gamma():
    for x in np.arange(0.1, 20.0, 0.1, dtype=float):
        assert is_close(gamma(x), float(mp.gamma(x)), relative=1e-12)


def test_ln_gamma():
    # whole numbers
    for x in np.arange(0.1, 1000.0, 10.0, dtype=float):
        assert is_close(ln_gamma(x), float(mp.loggamma(x)), relative=1e-12)

    # TODO: I ought to use deterministically random marks here instead of
    # those hacks

    # non-whole numbers
    for x in np.arange(0.1, 1000.0, 7.7039, dtype=float):
        assert is_close(ln_gamma(x), float(mp.loggamma(x)), relative=1e-12)
