from math import inf, exp, log, isfinite
from typing import Tuple
from ...rng import Rng


# x must be in [0, inf)
def interval_to_range(ratio: float, low: float, high: float):
    return low + (high - low) / (ratio + 1)


def _is_on_range(distribution) -> bool:
    return isfinite(distribution.lower) and isfinite(distribution.upper)


def _sample_rescale(low, high, distribution, rng: Rng):
    x = distribution.sample(rng)
    ratio = (x - distribution.lower) / (distribution.upper - distribution.lower)
    return interval_to_range(ratio, low, high)


def sample_range(low: float, high: float, distribution, rng: Rng) -> int | float:
    if _is_on_range(distribution):
        return _sample_rescale(low, high, distribution, rng)

    x = distribution.sample(rng)

    # if the distribution is full-line rescale it to positive numbers only
    if distribution.lower == -inf:
        x = exp(x)

    # fold lines and half-open intervals into a range
    new_point = interval_to_range(x, low, high)
    return new_point


def scale_on_range(
    low: float, high: float, distribution, rng: Rng
) -> Tuple[int | float, float]:
    """Pick a point on a range.

    The BEAST2 calculates the ratio and scaling as follows:

    - Choose a random number between `factor` and `1 / factor`.
    - The new value is `low + scale * (high - low)`.
    - The Hastings ratio is a logarithm of `scale`.

    For non-internal distributions this function does a similar thing: the
    scale is selected on `(0, inf)` (with exponentiation rescaling for
    full-line distributions) and the rest of the algorithm is the same as
    BEAST2.

    The distributions defined on an interval are rescaled to `(low, high)`.
    Since the target distribution is the same and we're just stretching it, I
    believe the Hastings ratio should always be 0.

    Args:
        low: The lowest possible point (TODO: inclusive?).
        high: The highest possible point (TODO: above).
        distribution: The distribution to sample the new point from.

    Returns:
        The tuple of new point and the Hastings ratio of the move.
    """

    if _is_on_range(distribution):
        return (_sample_rescale(low, high, distribution, rng), 0)

    scale = distribution.sample(rng)
    if distribution.lower == -inf:
        scale = exp(scale)
    new_point = interval_to_range(scale, low, high)
    return (new_point, log(scale))
