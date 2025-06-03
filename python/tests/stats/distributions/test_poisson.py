import pytest

from math import isclose, nan

from aspartik.stats.distributions import Poisson, PoissonError


def test_basic():
    d = Poisson(1)
    assert d.lambda_ == 1
    assert repr(d) == "Poisson(1)"
    assert isclose(0.367879441171442, d.pmf(1), rel_tol=1e-15)
    assert d.lower == 0
    assert d.upper == 2**64 - 1  # u64::MAX


def test_errors():
    with pytest.raises(ValueError) as error:
        Poisson(0)
    assert error.value.args[0] == PoissonError.LambdaInvalid

    with pytest.raises(ValueError) as error:
        Poisson(-1)
    assert error.value.args[0] == PoissonError.LambdaInvalid

    with pytest.raises(ValueError) as error:
        Poisson(nan)
    assert error.value.args[0] == PoissonError.LambdaInvalid
