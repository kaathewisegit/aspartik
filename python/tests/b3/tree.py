import pickle

from aspartik.b3 import Tree
from aspartik.rng import RNG


def test_pickle_roundtrip():
    old = Tree([str(i) for i in range(10)], RNG(4))
    new = pickle.loads(pickle.dumps(old))

    assert old.newick() == new.newick()


def test_pickle_state():
    rng = RNG(4)

    old = Tree([str(i) for i in range(10)], rng)
    internal = old.random_internal(rng)
    old.set_height(internal, 100)

    new = pickle.loads(pickle.dumps(old))

    assert old.newick() == new.newick()
    old.reject()
    new.reject()
    assert old.newick() == new.newick()


def test_other_child():
    rng = RNG(4)

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
