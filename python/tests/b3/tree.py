import pytest
from utils import random_float, random_msas, random_trees

from aspartik.b3.config import b3_config
from aspartik.b3.parameters import Tree
from aspartik.data.msa import MSA
from aspartik.rng import RNG


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


@pytest.mark.parametrize("msa", random_msas(10, 1000, num=20))
def test_dump_load_mcmc(msa: MSA, rng: RNG):
    mcmc = b3_config(
        msa,
        tree_prior="constant",
        substitution_model="JC",
        print_every=None,
    )
    mcmc.run(100)
    tree = mcmc.parameters[0]
    assert isinstance(tree, Tree)
    tree_state = tree.dump()

    deserialized_tree = Tree(msa.sequence_names(), rng)
    deserialized_tree.load(tree_state)

    assert tree.to_newick() == deserialized_tree.to_newick()


@pytest.mark.parametrize("pop_size", random_float(0, 1000, num=5))
def test_simulate_coalescent(pop_size: float, rng: RNG):
    for i in range(3, 1000):
        names = list(map(str, range(i)))
        heights = [rng.random_float(0, 100) for _ in range(i)]
        tree = Tree.simulate_coalescent(names, heights, pop_size, rng)
        tree.validate()
