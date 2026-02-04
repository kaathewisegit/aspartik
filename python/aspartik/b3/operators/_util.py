from math import exp, inf, isfinite

from ...rng import RNG
from ..parameters import Internal, Node, Tree


# x must be in [0, inf)
def interval_to_range(ratio: float, low: float, high: float):
    return low + (high - low) / (ratio + 1)


def _is_on_range(distribution) -> bool:
    return isfinite(distribution.lower) and isfinite(distribution.upper)


def _sample_rescale(low, high, distribution, rng: RNG):
    x = distribution.sample(rng)
    ratio = (x - distribution.lower) / (distribution.upper - distribution.lower)
    # `ratio` is in [0, 1]
    return low + (high - low) * ratio


def sample_range(low: float, high: float, distribution, rng: RNG) -> int | float:
    if _is_on_range(distribution):
        return _sample_rescale(low, high, distribution, rng)

    x = distribution.sample(rng)

    # if the distribution is full-line rescale it to positive numbers only
    if distribution.lower == -inf:
        x = exp(x)

    # fold lines and half-open intervals into a range
    new_point = interval_to_range(x, low, high)
    return new_point


def family(tree: Tree, rng: RNG) -> tuple[Node, Node, Internal, Internal | None]:
    root = tree.root

    node = tree.random_node(rng)
    while node == root:
        node = tree.random_node(rng)

    parent = tree.parent_of(node)
    assert parent is not None  # checked in the loop

    grandparent = tree.parent_of(parent)

    sibling = tree.other_child(parent, node)

    return node, sibling, parent, grandparent


def assert_two_internals(operator) -> None:
    if operator.tree.num_internals < 2:
        raise ValueError(
            f"`{operator.__class__.__name__}` requires the tree to have at least 2 internal nodes"
        )


def assert_factor(operator) -> None:
    if not (0.0 < operator.factor < 1.0):
        raise ValueError(f"Factor must be in range (0, 1), got {operator.factor}")
