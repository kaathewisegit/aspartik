from aspartik.b3.parameters import Real
from aspartik.distributions import LogNormal


def test_new():
    assert LogNormal(1, 1).scale == 1
    assert LogNormal(1.0, 1.0).scale == 1
    assert LogNormal(Real(1), 1.0).scale == 1
    assert LogNormal(1, Real(1.0)).scale == 1


def test_statitics():
    assert LogNormal(-2, 2).mean() == 1
    assert LogNormal(0, 1).median() == 1
    assert LogNormal(1, 1).mode() == 1
    assert isinstance(LogNormal(1, 1).variance(), float)
    assert isinstance(LogNormal(1, 1).entropy(), float)
