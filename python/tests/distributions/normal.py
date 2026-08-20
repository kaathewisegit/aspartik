from aspartik.b3.parameters import Real
from aspartik.distributions import Normal


def test_new():
    assert Normal(1, 1).mean == 1
    assert Normal(1.0, 1.0).mean == 1
    assert Normal(Real(1), 1.0).mean == 1
    assert Normal(1, Real(1.0)).mean == 1
