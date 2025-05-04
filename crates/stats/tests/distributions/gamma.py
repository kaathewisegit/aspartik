from aspartik.stats.distributions import Gamma, GammaError  # noqa: F401
from math import inf

try:
    Gamma(1, -2)
except ValueError as e:
    assert e.args[0] == GammaError.RateInvalid

g = Gamma(1, 2)
assert g.shape == 1
assert g.rate == 2
assert repr(g) == "Gamma(shape=1, rate=2)"
assert g.pdf(0.5) == 0.7357588823428847
assert g.lower == 0
assert g.upper == inf
