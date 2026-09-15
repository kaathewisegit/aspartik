import pytest

from aspartik.data.tree import BinaryTree, Tree, robinson_foulds_matrix
from aspartik.rng import RNG


def topology(tree):
    return [tree.children_of(node) for node in tree.internals()]


def indexed_tree(topology):
    tree = Tree()
    leaves = [tree.add_node(tree.root, str(index), 0.0) for index in range(4)]

    def attach(clade, parent):
        if isinstance(clade, int):
            tree.replace_parent(leaves[clade], parent)
        else:
            internal = tree.add_node(parent, length=0.0)
            for child in clade:
                attach(child, internal)

    for child in topology:
        attach(child, tree.root)
    return tree.to_binary()


def test_random_tree():
    tree = BinaryTree.random(100, RNG(4))

    assert tree.num_leaves == 100
    assert tree.num_nodes == 199
    assert tree.num_edges == 198
    assert len(tree.preorder()) == tree.num_nodes
    assert tree.preorder()[0] == tree.root
    assert sorted(tree.postorder()) == tree.nodes()
    assert all(tree.name(node) is None for node in tree.nodes())
    assert all(tree.node_metadata(node) is None for node in tree.nodes())
    assert all(tree.edge_length(child) == 0.0 for child in tree.edges())
    assert all(tree.edge_metadata(child) is None for child in tree.edges())


def test_random_tree_is_deterministic():
    first = BinaryTree.random(100, RNG(4))
    second = BinaryTree.random(100, RNG(4))
    assert topology(first) == topology(second)


def test_random_tree_advances_rng():
    rng = RNG(4)
    first = BinaryTree.random(100, rng)
    second = BinaryTree.random(100, rng)
    assert topology(first) != topology(second)


@pytest.mark.parametrize("num_leaves", [0, 1, 2**32])
def test_random_tree_rejects_invalid_sizes(num_leaves):
    with pytest.raises((OverflowError, RuntimeError)):
        BinaryTree.random(num_leaves, RNG(4))


def test_branch_score():
    first = BinaryTree.from_newick("(A:1,B:2);")
    second = BinaryTree.from_newick("(A:2,B:4);")

    assert first.branch_score(first) == 0.0
    assert first.branch_score(second) == pytest.approx(5.0**0.5)
    assert second.branch_score(first) == pytest.approx(5.0**0.5)


def test_triplet_distance():
    first = indexed_tree(((0, 1), (2, 3)))
    second = indexed_tree(((0, 2), (1, 3)))

    assert first.triplet_distance(first) == 0
    assert first.triplet_distance(second) == 4
    assert first.triplet_distance(second) == second.triplet_distance(first)


def test_triplet_distance_rejects_leaf_count_mismatch():
    first = BinaryTree.random(10, RNG(4))
    second = BinaryTree.random(11, RNG(5))

    with pytest.raises(RuntimeError, match="Expected both trees"):
        first.triplet_distance(second)


def test_robinson_foulds_matrix():
    trees = [
        indexed_tree(((0, 1), (2, 3))),
        indexed_tree((((0, 1), 2), 3)),
        indexed_tree(((0, 2), (1, 3))),
        indexed_tree(((1, 0), (3, 2))),
    ]
    assert robinson_foulds_matrix(trees) == [
        [0, 2, 4, 0],
        [2, 0, 4, 2],
        [4, 4, 0, 4],
        [0, 2, 4, 0],
    ]


def test_robinson_foulds_matrix_empty_and_single():
    assert robinson_foulds_matrix([]) == []
    tree = BinaryTree.random(10, RNG(4))
    assert robinson_foulds_matrix([tree]) == [[0]]


def test_robinson_foulds_matrix_rejects_leaf_count_mismatch():
    trees = [BinaryTree.random(10, RNG(4)), BinaryTree.random(11, RNG(5))]
    with pytest.raises(RuntimeError, match="Expected every tree to have 10 leaves"):
        robinson_foulds_matrix(trees)


def test_robinson_foulds_matrix_rejects_leaf_id_mismatch():
    trees = [
        BinaryTree.from_newick("((A:0,B:0):0,(C:0,D:0):0);"),
        BinaryTree.from_newick("((A:0,C:0):0,(B:0,D:0):0);"),
    ]
    with pytest.raises(RuntimeError, match="same leaf IDs"):
        robinson_foulds_matrix(trees)
