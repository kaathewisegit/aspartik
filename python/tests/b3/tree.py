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
    old.update_weight(internal, 100)

    new = pickle.loads(pickle.dumps(old))

    assert old.newick() == new.newick()
    old.reject()
    new.reject()
    assert old.newick() == new.newick()
