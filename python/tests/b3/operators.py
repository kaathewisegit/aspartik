from utils import check_tree_operator

from aspartik.b3.operators import (
    EpochScale,
    FixedHeightSubtreePruneRegraft,
    NarrowExchange,
    NodeSlide,
    ParamScale,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.stats.distributions import Uniform


def test_epoch_scale(rng):
    factory = lambda tree: EpochScale(tree, 0.9, Uniform(0, 1), rng, weight=1.0)
    check_tree_operator(factory)


def test_narrow_exchange(rng):
    factory = lambda tree: NarrowExchange(tree, rng)
    check_tree_operator(factory)


def test_wide_exchange(rng):
    factory = lambda tree: WideExchange(tree, rng)
    check_tree_operator(factory)


def test_wilson_balding(rng):
    factory = lambda tree: WilsonBalding(tree, rng)
    check_tree_operator(factory)


def test_node_slide(rng):
    factory = lambda tree: NodeSlide(tree, rng)
    check_tree_operator(factory)


def test_tree_scale(rng):
    factory = lambda tree: TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=1.0)
    check_tree_operator(factory)


def test_spr(rng):
    factory = lambda tree: FixedHeightSubtreePruneRegraft(tree, rng)
    check_tree_operator(factory)
