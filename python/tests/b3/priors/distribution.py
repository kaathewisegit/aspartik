from aspartik.b3.parameters import Real
from aspartik.b3.priors import Distribution
from aspartik.math import is_close
from aspartik.stats.distributions import Normal


def test_param():
    prior = Distribution(Real(2.0), Normal(0, 1))
    assert is_close(prior.probability(), -2.9189385332046727)
