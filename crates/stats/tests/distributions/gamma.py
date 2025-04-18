from stats.distributions import Gamma, GammaError

# catching classes that do not inherit from BaseException is not allowed
# try:
#     Gamma(1, -2)
# except GammaError as e:
#     print(e)

g = Gamma(1, 2)
assert g.shape == 1
assert g.rate == 2
assert repr(g) == "Gamma(1, 2)"
assert g.pdf(0.5) == 0.7357588823428847
