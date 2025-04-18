from math import isclose

from stats.distributions import Poisson

d = Poisson(1)
assert d.lambda_ == 1
assert repr(d) == "Poisson(1)"
assert isclose(0.367879441171442, d.pmf(1), rel_tol=1e-15)
