import pytest

from math import inf

from aspartik.stats.distributions import Gamma, GammaError


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
