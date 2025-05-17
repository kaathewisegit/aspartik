"""Utilities for basic statistics.

This library aims to mirror the functionality of `scipy.stats`.  The reason the
latter could not be used is because SciPy is optimized for batch processing,
which causes it to be quite slow for singular calls, like creating a new
distribution and calculating a PDF of one value.  `b3` operators and priors
have this exact usage pattern[^params], so `stats` was created as a library
which performs reasonably well on such workloads.


[^params]:
    Since each MCMC step depends on the last one priors have to be recalculated
    on each step.  And operators also only propose one move at a time.
    Additionally, since parameters can change, it's easier to recreate
    distribution classes on each step, but SciPy ones were definitely not
    intended for that.
"""

from . import distributions as distributions
