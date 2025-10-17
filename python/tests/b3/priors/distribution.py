import pytest

from aspartik.b3 import Tree
from aspartik.b3.parameters import Real, Root
from aspartik.b3.priors import Distribution
from aspartik.math import is_close
from aspartik.stats.distributions import Normal, Poisson


def test_float():
    prior = Distribution(2.0, Normal(0, 1))
    assert is_close(prior.probability(), -2.9189385332046727)


def test_param():
    prior = Distribution(Real(2.0), Normal(0, 1))
    assert is_close(prior.probability(), -2.9189385332046727)


def test_root_param(rng):
    tree = Tree(["seq0", "seq1"], rng)
    tree.set_height(tree.root, 2)
    tree.accept()

    param = Root(tree)

    prior = Distribution(param, Normal(0, 1))
    assert is_close(prior.probability(), -2.9189385332046727)
