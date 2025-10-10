import pytest

from aspartik.b3 import Integer, Real
from aspartik.b3.priors import Distribution
from aspartik.stats.distributions import Normal, Poisson


def test_param_type():
    with pytest.raises(TypeError):
        Distribution(Integer(1), Normal(0, 1))

    with pytest.raises(TypeError):
        Distribution(Real(1), Poisson(1))
