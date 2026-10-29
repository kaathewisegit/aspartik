import dendropy
import pytest
from dendropy.calculate import treecompare
from utils import random_integer

from array import array
from pathlib import Path

from aspartik.data import TaxonSet
from aspartik.data.tree import (
    BinaryTree,
    TreeBuilder,
    branch_score_matrix,
    robinson_foulds_matrix,
    triplet_distance_matrix,
)
from aspartik.rng import RNG


def topology(tree):
    return [tree.children_of(node) for node in tree.internals()]


def indexed_tree(topology):
    tree = TreeBuilder()
    root = tree.add_dummy_node()
    tree.set_root(root)
    leaves = [tree.add_node(root, str(index), 0.0) for index in range(4)]
    tree.set_node_children(root, [])

    def attach(clade, parent):
        if isinstance(clade, int):
            leaf = leaves[clade]
            tree.set_node_parent(leaf, parent)
            return leaf
        internal = tree.add_node(parent, length=0.0)
        children = [attach(child, internal) for child in clade]
        tree.set_node_children(internal, children)
        return internal

    tree.set_node_children(root, [attach(child, root) for child in topology])
    return tree.to_binary()


def test_random_tree():
    taxa = TaxonSet.ranged_ints(100)
    tree = BinaryTree.random(taxa, RNG(4))

    assert tree.num_leaves == 100
    assert tree.num_nodes == 199
    assert tree.num_edges == 198
    assert len(tree.preorder()) == tree.num_nodes
    assert tree.preorder()[0] == tree.root
    assert tree.taxa == taxa
    assert [tree.name(leaf) for leaf in tree.leaves()] == list(taxa)
    assert all(tree.name(node) is None for node in tree.internals())
    assert all(tree.edge_length(child) == 1.0 for child in tree.edges())


def test_random_tree_is_deterministic():
    taxa = TaxonSet.ranged_ints(100)
    first = BinaryTree.random(taxa, RNG(4))
    second = BinaryTree.random(taxa, RNG(4))
    assert topology(first) == topology(second)


def test_random_tree_advances_rng():
    rng = RNG(4)
    taxa = TaxonSet.ranged_ints(100)
    first = BinaryTree.random(taxa, rng)
    second = BinaryTree.random(taxa, rng)
    assert topology(first) != topology(second)


@pytest.mark.parametrize("num_leaves", [0, 1])
def test_random_tree_rejects_invalid_sizes(num_leaves):
    with pytest.raises(RuntimeError):
        BinaryTree.random(TaxonSet.ranged_ints(num_leaves), RNG(4))


def test_branch_score():
    first = BinaryTree.from_newick("(A:1,B:2);")
    second = BinaryTree.from_newick("(A:2,B:4);")

    assert first.branch_score(first) == 0.0
    assert first.branch_score(second) == pytest.approx(5.0**0.5)
    assert second.branch_score(first) == pytest.approx(5.0**0.5)


def test_branch_score_matrix():
    trees = [
        BinaryTree.from_newick("(A:1,B:2);"),
        BinaryTree.from_newick("(A:2,B:4);"),
        BinaryTree.from_newick("(A:0,B:0);"),
    ]
    matrix = branch_score_matrix(trees)
    assert isinstance(matrix, array)
    assert matrix.typecode == "d"
    assert list(matrix) == pytest.approx(
        [first.branch_score(second) for first in trees for second in trees]
    )
    assert branch_score_matrix([]) == array("d", [])
    assert branch_score_matrix(trees[:1]) == array("d", [0.0])


def test_branch_score_matrix_rejects_leaf_mismatch():
    first = BinaryTree.from_newick("(A:1,B:2);")
    different_count = BinaryTree.from_newick("((A:1,B:2):1,C:3);")
    different_name = BinaryTree.from_newick("(A:1,C:2);")
    with pytest.raises(RuntimeError, match="Expected every tree to have 2 leaves"):
        branch_score_matrix([first, different_count])
    with pytest.raises(RuntimeError, match="same leaf IDs"):
        branch_score_matrix([first, different_name])


def test_triplet_distance():
    first = indexed_tree(((0, 1), (2, 3)))
    second = indexed_tree(((0, 2), (1, 3)))

    assert first.triplet_distance(first) == 0
    assert first.triplet_distance(second) == 4
    assert first.triplet_distance(second) == second.triplet_distance(first)


def test_triplet_distance_rejects_leaf_count_mismatch():
    first = BinaryTree.random(TaxonSet.ranged_ints(10), RNG(4))
    second = BinaryTree.random(TaxonSet.ranged_ints(11), RNG(5))

    with pytest.raises(RuntimeError, match="Expected both trees"):
        first.triplet_distance(second)


def test_triplet_distance_matrix():
    trees = [
        indexed_tree(((0, 1), (2, 3))),
        indexed_tree(((0, 2), (1, 3))),
        indexed_tree((((0, 1), 2), 3)),
    ]
    expected = array(
        "Q",
        [first.triplet_distance(second) for first in trees for second in trees],
    )
    assert triplet_distance_matrix(trees) == expected
    assert triplet_distance_matrix(trees, 1) == expected
    assert triplet_distance_matrix([]) == array("Q", [])
    assert triplet_distance_matrix(trees[:1]) == array("Q", [0])


def test_triplet_distance_matrix_rejects_leaf_count_mismatch():
    trees = [
        BinaryTree.random(TaxonSet.ranged_ints(10), RNG(4)),
        BinaryTree.random(TaxonSet.ranged_ints(11), RNG(5)),
    ]
    with pytest.raises(RuntimeError, match="Expected every tree to have 10 leaves"):
        triplet_distance_matrix(trees)


def test_triplet_distance_matrix_rejects_leaf_id_mismatch():
    trees = [
        BinaryTree.from_newick("((A:0,B:0):0,(C:0,D:0):0);"),
        BinaryTree.from_newick("((A:0,C:0):0,(B:0,D:0):0);"),
    ]
    with pytest.raises(RuntimeError, match="same leaf IDs"):
        triplet_distance_matrix(trees)


def test_robinson_foulds_matrix():
    trees = [
        indexed_tree(((0, 1), (2, 3))),
        indexed_tree((((0, 1), 2), 3)),
        indexed_tree(((0, 2), (1, 3))),
        indexed_tree(((1, 0), (3, 2))),
    ]
    # fmt: off
    assert robinson_foulds_matrix(trees) == array(
        "I",
        [
            0, 2, 4, 0,
            2, 0, 4, 2,
            4, 4, 0, 4,
            0, 2, 4, 0,
        ],
    )


def test_robinson_foulds_matrix_empty_and_single():
    assert robinson_foulds_matrix([]) == array("I", [])
    tree = BinaryTree.random(TaxonSet.ranged_ints(10), RNG(4))
    assert robinson_foulds_matrix([tree]) == array("I", [0])


def test_robinson_foulds_matrix_rejects_leaf_count_mismatch():
    trees = [
        BinaryTree.random(TaxonSet.ranged_ints(10), RNG(4)),
        BinaryTree.random(TaxonSet.ranged_ints(11), RNG(5)),
    ]
    with pytest.raises(RuntimeError, match="Expected every tree to have 10 leaves"):
        robinson_foulds_matrix(trees)


def test_robinson_foulds_matrix_rejects_leaf_id_mismatch():
    trees = [
        BinaryTree.from_newick("((A:0,B:0):0,(C:0,D:0):0);"),
        BinaryTree.from_newick("((A:0,C:0):0,(B:0,D:0):0);"),
    ]
    with pytest.raises(RuntimeError, match="same leaf IDs"):
        robinson_foulds_matrix(trees)


def test_robinson_foulds(rng: RNG):
    # 4 taxa
    taxa = TaxonSet.ranged_ints(4)

    # identical -> 0
    tree = BinaryTree.from_newick("((0:0,1:0):0,(2:0,3:0):0);", taxa)
    other = BinaryTree.from_newick("((0:0,1:0):0,(2:0,3:0):0);", taxa)
    assert tree.robinson_foulds(other) == 0

    # shared clade {0,1}, differ on the other -> RF 2
    other = BinaryTree.from_newick("(((0:0,1:0):0,2:0):0,3:0);", taxa)
    assert tree.robinson_foulds(other) == 2

    # no shared non-trivial clades -> RF 4 (maximum for 4 taxa)
    other = BinaryTree.from_newick("((0:0,2:0):0,(1:0,3:0):0);", taxa)
    assert tree.robinson_foulds(other) == 4

    # 5 taxa
    taxa = TaxonSet.ranged_ints(5)

    # identical
    tree = BinaryTree.from_newick("((0:0,1:0):0,(2:0,(3:0,4:0):0):0);", taxa)
    other = BinaryTree.from_newick("((0:0,1:0):0,(2:0,(3:0,4:0):0):0);", taxa)
    assert tree.robinson_foulds(other) == 0

    # one NNI move: {0,1} and {3,4} shared -> RF 2
    other = BinaryTree.from_newick("((0:0,1:0):0,((2:0,3:0):0,4:0):0);", taxa)
    assert tree.robinson_foulds(other) == 2

    # caterpillar: only {0,1} shared -> RF 4
    other = BinaryTree.from_newick("((((0:0,1:0):0,2:0):0,3:0):0,4:0);", taxa)
    assert tree.robinson_foulds(other) == 4

    # maximally different (left comb vs right comb)
    tree = BinaryTree.from_newick("(0:0,(1:0,(2:0,(3:0,4:0):0):0):0);", taxa)
    other = BinaryTree.from_newick("((((0:0,1:0):0,2:0):0,3:0):0,4:0);", taxa)
    assert tree.robinson_foulds(other) == 6

    # 6 taxa
    taxa = TaxonSet.ranged_ints(6)
    tree = BinaryTree.from_newick("(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);", taxa)
    other = BinaryTree.from_newick("(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);", taxa)
    assert tree.robinson_foulds(other) == 0

    # reroot: {0,1} and {2,3} shared, root clade differs
    other = BinaryTree.from_newick("((0:0,1:0):0,((2:0,3:0):0,(4:0,5:0):0):0);", taxa)
    assert tree.robinson_foulds(other) == 2

    # symmetrical
    assert tree.robinson_foulds(other) == other.robinson_foulds(tree)

    # 8 taxa
    balanced = "((((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0):0,(6:0,7:0):0);"
    taxa = TaxonSet.ranged_ints(8)
    tree = BinaryTree.from_newick(balanced, taxa)
    other = BinaryTree.from_newick(balanced, taxa)
    assert tree.robinson_foulds(other) == 0


@pytest.mark.parametrize("size", random_integer(3, 1_000))
def test_robinson_foulds_sim(size: int, rng: RNG):
    taxa = TaxonSet.ranged_ints(size)
    a = BinaryTree.random(taxa, rng)
    b = BinaryTree.random(taxa, rng)

    def dendropy_distance(a, b):
        tns = dendropy.TaxonNamespace()
        a = dendropy.Tree.get(data=a, schema="newick", taxon_namespace=tns)
        b = dendropy.Tree.get(data=b, schema="newick", taxon_namespace=tns)
        # TODO: leaf labels
        a.is_rooted = True
        b.is_rooted = True

        a.encode_bipartitions()
        b.encode_bipartitions()

        return treecompare.symmetric_difference(a, b)

    assert a.robinson_foulds(b) == dendropy_distance(a.to_newick(), b.to_newick())


@pytest.mark.parametrize(
    "path",
    sorted(Path("data/runs").glob("*/beast*.trees")),
    ids=lambda path: path.parent.name,
)
def test_robinson_foulds_matrix_on_beast_trees(path):
    builder = TreeBuilder()
    newicks = path.read_text().splitlines()
    builder.load_newick(newicks[0])
    first = builder.to_binary()
    taxa = first.taxa

    trees = [first]
    for newick in newicks[1:]:
        builder.load_newick(newick)
        trees.append(builder.to_binary(taxa))

    matrix = robinson_foulds_matrix(trees)

    assert trees
    assert len(matrix) == len(trees) ** 2
    assert all(matrix[index * len(trees) + index] == 0 for index in range(len(trees)))
