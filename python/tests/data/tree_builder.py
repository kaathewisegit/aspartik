import pytest

from aspartik.data.tree import TreeBuilder


def test_from_newick():
    tree = TreeBuilder.from_newick("(A:1.0,B:2.0)C:3.0;")

    root = tree.root
    assert root is not None
    assert tree.name(root) == "C"
    assert tree.edge_length(root) == 3.0

    children = tree.children_of(root)
    assert len(children) == 2
    assert [tree.name(child) for child in children] == ["A", "B"]
    assert [tree.edge_length(child) for child in children] == [1.0, 2.0]
    assert all(tree.parent_of(child) == root for child in children)

    assert tree.num_nodes == 3
    assert len(tree) == 3
    assert tree.is_binary()


# TODO: tree fixtures
def test_from_newick_roundtrip():
    newick = "(A:1,B:2);"
    assert TreeBuilder.from_newick(newick).to_newick() == newick


def test_from_newick_metadata():
    tree = TreeBuilder.from_newick("(A[&node]:1.0[&edge])C[&root];")

    root = tree.root
    assert root is not None
    assert tree.node_metadata(root) == "&root"

    (leaf,) = tree.children_of(root)
    assert tree.name(leaf) == "A"
    assert tree.node_metadata(leaf) == "&node"
    assert tree.edge_metadata(leaf) == "&edge"


def test_from_newick_rejects_invalid():
    with pytest.raises(Exception):
        TreeBuilder.from_newick("(A,B")


def test_add_node_and_dummy():
    tree = TreeBuilder()
    assert tree.num_nodes == 1
    root = tree.root

    child = tree.add_node(root, "A", 1.0)
    assert tree.parent_of(child) == root
    assert tree.children_of(root) == [child]
    assert tree.name(child) == "A"
    assert tree.edge_length(child) == 1.0
    assert tree.is_leaf(child)
    assert not tree.is_leaf(root)
    assert tree.num_nodes == 2


def test_root():
    tree = TreeBuilder.from_newick("(A:1.0);")
    root = tree.root
    assert root is not None

    assert tree.parent_of(root) is None
    assert tree.edge_length(root) is None


def test_setters():
    tree = TreeBuilder()
    root = tree.root
    child = tree.add_node(root, "A", 1.0)

    tree.set_name(child, "B")
    tree.set_edge_length(child, 2.5)
    tree.set_node_metadata(child, "&x")
    tree.set_edge_metadata(child, "&y")

    assert tree.name(child) == "B"
    assert tree.edge_length(child) == 2.5
    assert tree.node_metadata(child) == "&x"
    assert tree.edge_metadata(child) == "&y"


def test_set_node_children():
    tree = TreeBuilder()
    root = tree.root
    left = tree.add_node(root, "A", 1.0)
    right = tree.add_node(root, "B", 2.0)
    tree.set_root(root)

    tree.set_node_children(root, [right, left])
    assert tree.children_of(root) == [right, left]
    assert tree.is_binary()

    tree.to_binary()


def test_to_binary():
    tree = TreeBuilder.from_newick("((A:1,B:2):3,C:4);")
    binary = tree.to_binary()

    assert binary.num_leaves == 3
    assert binary.num_nodes == 5
    assert binary.num_edges == 4
    assert [binary.name(leaf) for leaf in binary.leaves()] == ["A", "B", "C"]
