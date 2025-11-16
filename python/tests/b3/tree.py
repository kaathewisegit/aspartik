import pytest
from utils import random_trees

import pickle

from aspartik.b3 import Tree
from aspartik.rng import RNG


def test_pickle_roundtrip(rng):
    old = Tree([str(i) for i in range(10)], rng)
    new = pickle.loads(pickle.dumps(old))

    assert old.newick() == new.newick()


def test_pickle_state(rng):
    old = Tree([str(i) for i in range(10)], rng)
    internal = old.random_internal(rng)
    old.set_height(internal, 100)

    new = pickle.loads(pickle.dumps(old))

    assert old.newick() == new.newick()
    old.reject()
    new.reject()
    assert old.newick() == new.newick()


def test_other_child(rng):
    tree = Tree(["1", "2"], rng)

    n1 = tree.leaf_by_name("1")
    n2 = tree.leaf_by_name("2")
    root = tree.root
    assert n1 is not None
    assert n2 is not None

    other = tree.other_child(root, n1)
    assert other == n2

    other = tree.other_child(root, n2)
    assert other == n1


def test_total_length(rng):
    tree = Tree(["1", "2"], rng)
    tree.set_height(tree.root, 10)

    assert tree.total_length() == 20


def test_names(rng):
    tree = Tree(["1", "2"], rng)
    assert tree.names == ["1", "2"]

    names = [str(rng.random_int(0, 2**32)) for _ in range(1000)]
    tree = Tree(names, rng)
    assert tree.names == names


@pytest.mark.parametrize("tree", random_trees(10, 20, num=15))
def test_swap_parents(tree, rng):
    a, a_parent = tree.random_nonroot_node(rng)
    b, b_parent = tree.random_nonroot_node(rng)

    a_invalid = not tree.height_of(a) < tree.height_of(b_parent)
    b_invalid = not tree.height_of(b) < tree.height_of(a_parent)

    if a_invalid or b_invalid:
        with pytest.raises(RuntimeError):
            tree.swap_parents(a, b)
        return
    else:
        tree.swap_parents(a, b)

    new_a_parent, new_b_parent = tree.parent_of(a), tree.parent_of(b)

    assert a_parent == new_b_parent
    assert b_parent == new_a_parent
