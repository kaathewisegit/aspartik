from utils import check_tree_operator

from aspartik.b3.operators import (
    FixedHeightSPR,
    NarrowExchange,
    NodeSlide,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.stats.distributions import Uniform


def test_narrow_exchange(rng):
    check_tree_operator(lambda tree: NarrowExchange(tree, rng))


def test_wide_exchange(rng):
    check_tree_operator(lambda tree: WideExchange(tree, rng))


def test_wilson_balding(rng):
    check_tree_operator(lambda tree: WilsonBalding(tree, rng))


def test_node_slide(rng):
    check_tree_operator(lambda tree: NodeSlide(tree, rng))


def test_tree_scale(rng):
    check_tree_operator(
        lambda tree: TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=1.0)
    )


def test_spr(rng):
    check_tree_operator(lambda tree: FixedHeightSPR(tree, rng))
