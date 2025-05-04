from ...rng import Rng
from math import inf, exp


# x must be in [0, inf)
def interval_to_range(ratio, low, high):
    return low + (high - low) / (ratio + 1)


def rescale_range(x, pre_low, pre_high, low, high):
    ratio = (x - pre_low) / (pre_high - pre_low)
    return interval_to_range(ratio, low, high)


def sample_range(low, high, distribution, rng: Rng):
    x = distribution.sample(rng)

    # full line distribution
    if distribution.lower == -inf:
        x = exp(x)

    # lines and half-open intervals
    if distribution.upper == inf:
        x = interval_to_range(x, low, high)
        return x

    # the distribution is on a range
    return rescale_range(x, distribution.lower, distribution.upper, low, high)
