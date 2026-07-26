import dendropy
import pytest
from dendropy.calculate import treecompare
from utils import random_integer, random_msas, random_trees

import itertools

from aspartik.b3.config import MCMCConfig
from aspartik.b3.parameters import Tree
from aspartik.data.msa import MSA
from aspartik.data.newick import Tree as NewickTree
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
    mcmc = MCMCConfig(
        msa,
        tree_prior="constant",
        substitution_model="JC",
        print_every=None,
    ).b3_mcmc()
    mcmc.run(100)
    tree = mcmc.parameters[2]
    assert isinstance(tree, Tree)
    tree_state = tree.dump()

    deserialized_tree = Tree(msa.sequence_names(), rng)
    deserialized_tree.load(tree_state)

    assert tree.to_newick() == deserialized_tree.to_newick()


def test_ola(rng: RNG):
    # Paper: https://arxiv.org/abs/2509.16405v1
    # figure 1
    tree = Tree([str(i) for i in range(4)], rng)

    newick = NewickTree("(((0:0,1:0):0,3:0):0,2:0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, -1, -1]

    newick = NewickTree("(((0:0,2:0):0,3:0):0,1:0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, 0, -2]

    newick = NewickTree("(((1:0,2:0):0,3:0):0,0:0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, 1, -2]

    # figure 2
    tree = Tree([str(i) for i in range(6)], rng)

    newick = NewickTree("(((0:0,(1:0,5:0):0):0,(3:0,4:0):0):0,2:0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, -1, -1, 3, 1]

    newick = NewickTree("((0:0,1:0):0,(((5:0,3:0):0,4:0):0,2:0):0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, -1, 2, 3, 3]

    # Paper: https://doi.org/10.1007/s11538-026-01611-9
    tree = Tree([str(i) for i in range(5)], rng)

    # figure 1
    newick = NewickTree("((0:0,(2:0,3:0):0):0,(1:0,4:0):0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, 0, 2, 1]

    # figure 2
    newick = NewickTree("((0:0,((1:0,3:0):0,4:0):0):0,2:0);")
    tree.load_newick(newick)
    assert tree.ola() == [0, -1, 1, -3]


def test_mrca(rng: RNG):
    tree = Tree([str(i) for i in range(4)], rng)
    newick = NewickTree("((0:0,1:0):0,(2:0,3:0):0);")
    tree.load_newick(newick)

    leaf0, leaf1, leaf2, leaf3 = tree.leaves()

    mrca_01 = tree.parent_of(leaf0)
    assert mrca_01 is not None

    mrca_23 = tree.parent_of(leaf2)
    assert mrca_23 is not None

    for a, b in itertools.permutations({leaf0, leaf1, mrca_01}, 2):
        assert mrca_01 == tree.mrca(a, b)

    for a, b in itertools.permutations({leaf2, leaf3, mrca_23}, 2):
        assert mrca_23 == tree.mrca(a, b)

    assert tree.mrca(leaf0, leaf2) == tree.root
    assert tree.mrca(leaf1, leaf3) == tree.root


def test_robinson_foulds(rng: RNG):
    # 4 taxa
    tree = Tree([str(i) for i in range(4)], rng)
    other = Tree([str(i) for i in range(4)], rng)

    # identical -> 0
    tree.load_newick(NewickTree("((0:0,1:0):0,(2:0,3:0):0);"))
    other.load_newick(NewickTree("((0:0,1:0):0,(2:0,3:0):0);"))
    assert tree.robinson_foulds(other) == 0

    # shared clade {0,1}, differ on the other -> RF 2
    other.load_newick(NewickTree("(((0:0,1:0):0,2:0):0,3:0);"))
    assert tree.robinson_foulds(other) == 2

    # no shared non-trivial clades -> RF 4 (maximum for 4 taxa)
    other.load_newick(NewickTree("((0:0,2:0):0,(1:0,3:0):0);"))
    assert tree.robinson_foulds(other) == 4

    # 5 taxa
    tree = Tree([str(i) for i in range(5)], rng)
    other = Tree([str(i) for i in range(5)], rng)

    # identical
    tree.load_newick(NewickTree("((0:0,1:0):0,(2:0,(3:0,4:0):0):0);"))
    other.load_newick(NewickTree("((0:0,1:0):0,(2:0,(3:0,4:0):0):0);"))
    assert tree.robinson_foulds(other) == 0

    # one NNI move: {0,1} and {3,4} shared -> RF 2
    other.load_newick(NewickTree("((0:0,1:0):0,((2:0,3:0):0,4:0):0);"))
    assert tree.robinson_foulds(other) == 2

    # caterpillar: only {0,1} shared -> RF 4
    other.load_newick(NewickTree("((((0:0,1:0):0,2:0):0,3:0):0,4:0);"))
    assert tree.robinson_foulds(other) == 4

    # maximally different (left comb vs right comb)
    tree.load_newick(NewickTree("(0:0,(1:0,(2:0,(3:0,4:0):0):0):0);"))
    other.load_newick(NewickTree("((((0:0,1:0):0,2:0):0,3:0):0,4:0);"))
    assert tree.robinson_foulds(other) == 6

    # 6 taxa
    tree = Tree([str(i) for i in range(6)], rng)
    other = Tree([str(i) for i in range(6)], rng)

    tree.load_newick(NewickTree("(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);"))
    other.load_newick(NewickTree("(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);"))
    assert tree.robinson_foulds(other) == 0

    # reroot: {0,1} and {2,3} shared, root clade differs
    other.load_newick(NewickTree("((0:0,1:0):0,((2:0,3:0):0,(4:0,5:0):0):0);"))
    assert tree.robinson_foulds(other) == 2

    # symmetrical
    assert tree.robinson_foulds(other) == other.robinson_foulds(tree)

    # 8 taxa
    tree = Tree([str(i) for i in range(8)], rng)
    other = Tree([str(i) for i in range(8)], rng)

    balanced = "((((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0):0,(6:0,7:0):0);"
    tree.load_newick(NewickTree(balanced))
    other.load_newick(NewickTree(balanced))
    assert tree.robinson_foulds(other) == 0


@pytest.mark.parametrize("size", random_integer(3, 1_000))
def test_robinson_foulds_sim(size: int, rng: RNG):
    a = Tree([str(i) for i in range(size)], rng)
    b = Tree([str(i) for i in range(size)], rng)

    def dendropy_distance(a, b):
        tns = dendropy.TaxonNamespace()
        a = dendropy.Tree.get(data=a, schema="newick", taxon_namespace=tns)
        b = dendropy.Tree.get(data=b, schema="newick", taxon_namespace=tns)
        a.is_rooted = True
        b.is_rooted = True

        a.encode_bipartitions()
        b.encode_bipartitions()

        return treecompare.symmetric_difference(a, b)

    assert a.robinson_foulds(b) == dendropy_distance(a.to_newick(), b.to_newick())
