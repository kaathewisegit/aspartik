from utils import check_tree_operator

from aspartik.b3.operators import (
    EpochScale,
    FixedHeightSPR,
    NarrowExchange,
    NodeSlide,
    TreeScale,
    WideExchange,
    WilsonBalding,
)
from aspartik.b3.parameters import Tree
from aspartik.stats.distributions import Uniform


def test_epoch_scale(rng):
    def factory(tree: Tree):
        return EpochScale(tree, 0.9, Uniform(0, 1), rng, weight=1.0)

    check_tree_operator(factory)


def test_narrow_exchange(rng):
    def factory(tree: Tree):
        return NarrowExchange(tree, rng)

    check_tree_operator(factory)


def test_wide_exchange(rng):
    def factory(tree: Tree):
        return WideExchange(tree, rng)

    check_tree_operator(factory)


def test_wilson_balding(rng):
    def factory(tree: Tree):
        return WilsonBalding(tree, rng)

    check_tree_operator(factory)


def test_node_slide(rng):
    def factory(tree: Tree):
        return NodeSlide(tree, rng)

    check_tree_operator(factory)


def test_tree_scale(rng):
    def factory(tree: Tree):
        return TreeScale(tree, 0.9, Uniform(0, 1), rng, weight=1.0)

    check_tree_operator(factory)


def test_spr(rng):
    def factory(tree: Tree):
        return FixedHeightSPR(tree, rng)

    check_tree_operator(factory)
