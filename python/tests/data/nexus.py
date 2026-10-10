from aspartik.data.tree import (
    read_binary_tree_str,
    read_binary_trees_path,
)

NEXUS = """#NEXUS

BEGIN TREES;
	TRANSLATE
		1 Taxon1,
		2 Taxon2,
		3 Taxon3,
		4 Taxon4;
	TREE tree1 = [&R] (((1:0.1, 2:0.2):0.3, 3:0.4):0.5, 4:0.6);
	TREE tree2 = [&R] ((1:0.2, (2:0.1, 3:0.3):0.2):0.4, 4:0.5);
END;
"""

EXPECTED = [
    "(((Taxon1:0.1,Taxon2:0.2):0.3,Taxon3:0.4):0.5,Taxon4:0.6);",
    "((Taxon1:0.2,(Taxon2:0.1,Taxon3:0.3):0.2):0.4,Taxon4:0.5);",
]


def test_read_binary_tree_str():
    trees = read_binary_tree_str(NEXUS)
    assert [str(tree) for tree in trees] == EXPECTED
    assert [tree.num_leaves for tree in trees] == [4, 4]


def test_read_binary_trees_path(tmp_path):
    path = tmp_path / "trees.nex"
    path.write_text(NEXUS)
    trees = read_binary_trees_path(path)
    assert [str(tree) for tree in trees] == EXPECTED
