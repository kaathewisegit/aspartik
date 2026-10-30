use anyhow::Result;
use arbitrary::Unstructured;
use arbtest::arbtest;

use std::{
	fs,
	io::{BufRead, BufReader},
	path::PathBuf,
};

use data::tree::{BinaryTree, Tree, TreeBuilder};

#[test]
fn roundtrip_basic() {
	let s = "(A:0.1,B:0.2,(C:0.3,D:0.4)E:0.5)F;";
	let tree = TreeBuilder::from_newick(s).unwrap();
	assert_eq!(s, tree.to_newick());
}

#[test]
fn labels_lengths_and_empty_fields() {
	for (source, expected) in [
		(
			"(A:0.1,B:2e-3,(C,D)CD:4E+2)ROOT;",
			"(A:0.1,B:0.002,(C,D)CD:400)ROOT;",
		),
		("(,A,,)ROOT;", "(,A,,)ROOT;"),
	] {
		let tree = TreeBuilder::from_newick(source).unwrap();
		assert_eq!(tree.to_newick(), expected);
	}
}

#[test]
fn quoted_labels() -> Result<()> {
	let escaped = TreeBuilder::from_newick("('Prince''s':0.1,'');")?;
	assert_eq!(escaped.to_newick(), "('Prince''s':0.1,);");
	assert_eq!(escaped.name_of(escaped.root().unwrap()), "",);
	let children = escaped.children_of(escaped.root().unwrap());
	assert_eq!(children.len(), 2);
	assert_eq!(escaped.name_of(children[0]), "Prince's");
	assert_eq!(escaped.name_of(children[1]), "");

	let doubled = TreeBuilder::from_newick("('a''''b':0.1);")?;
	let node = doubled.children_of(doubled.root().unwrap())[0];
	assert_eq!(doubled.name_of(node), "a''b");
	assert_eq!(doubled.to_newick(), "('a''''b':0.1);");

	Ok(())
}

#[test]
fn multifurcating_roundtrip() -> Result<()> {
	let source = "(A:1,B:2,C:3,D:4)ROOT;";
	let builder = TreeBuilder::from_newick(source)?;
	assert!(!builder.is_binary());
	assert!(builder.to_binary().is_err());
	assert_eq!(builder.children_of(builder.root().unwrap()).len(), 4);
	assert_eq!(builder.to_newick(), source);

	Ok(())
}

#[test]
fn malformed_and_truncated_input() {
	for source in [
		"",
		"(A:1,B:1)",
		"(A:1,B:1;",
		"A:wat;",
		"A:1:2;",
		"('A:1,B:1);",
		"(A[&x=1,B:1);",
		"(A:1,B:1));",
		"(A:1,B:1); trailing",
	] {
		assert!(
			TreeBuilder::from_newick(source).is_err(),
			"accepted {source:?}"
		);
	}
}

fn nested_tree(depth: usize) -> String {
	let mut source = String::with_capacity(depth * 2 + 2);
	for _ in 0..depth {
		source.push('(');
	}
	source.push('A');
	for _ in 0..depth {
		source.push(')');
	}
	source.push(';');
	source
}

#[test]
fn eight_million_nested_tree_does_not_exhaust_the_stack() -> Result<()> {
	const DEPTH: usize = 200;
	let tree = TreeBuilder::from_newick(&nested_tree(DEPTH))?;
	assert_eq!(tree.num_nodes(), (DEPTH + 1));

	Ok(())
}

#[test]
fn deep_ladder_parse_and_write() -> Result<()> {
	const NUM_LEAVES: usize = 5_000;
	let mut source = String::with_capacity(NUM_LEAVES * 16);
	for leaf in 0..NUM_LEAVES - 1 {
		source.push('(');
		source.push_str(&format!("L{leaf}:1,"));
	}
	source.push_str(&format!("L{}:1", NUM_LEAVES - 1));
	for _ in 0..NUM_LEAVES - 1 {
		source.push_str("):1");
	}
	source.push(';');

	let tree = TreeBuilder::from_newick(&source)?;
	assert_eq!(tree.num_nodes(), NUM_LEAVES * 2 - 1);
	let output = tree.to_newick();
	let reparsed = TreeBuilder::from_newick(&output)?;
	assert_eq!(reparsed.num_nodes(), tree.num_nodes());
	assert!(reparsed.is_binary());

	Ok(())
}

fn arbitrary_builder(
	u: &mut Unstructured<'_>,
	num_leaves: u32,
) -> arbitrary::Result<TreeBuilder> {
	let mut tree = TreeBuilder::new();
	let root = tree.add_dummy_node();
	tree.set_root(root);
	tree.set_node_name(root, "root");
	tree.set_node_attribute(root, "&id=0");
	let mut leaves = vec![root];
	while leaves.len() < num_leaves as usize {
		let index = u.int_in_range(0..=leaves.len() - 1)?;
		let parent = leaves.swap_remove(index);
		for _ in 0..2 {
			let id = tree.num_nodes() as u32;
			let child = tree.add_node(
				&format!("node {id}"),
				f64::from(id) / 10.0,
				parent,
			);
			tree.set_node_attribute(child, &format!("&id={id}"));
			tree.set_edge_attribute(child, &format!("&edge={id}"));
			leaves.push(child);
		}
	}
	Ok(tree)
}

#[test]
fn arbitrary_roundtrip() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=10_000)?;
		let tree = arbitrary_builder(u, num_leaves)?;
		let first = tree.to_newick();
		let reparsed = TreeBuilder::from_newick(&first).unwrap();
		let second = reparsed.to_newick();
		assert_eq!(first, second);
		let binary: BinaryTree = reparsed.to_binary().unwrap();
		assert_eq!(binary.num_leaves(), num_leaves);
		Ok(())
	})
	.size_min(2u32.pow(20));
}

#[test]
fn parse_runs() {
	let mut runs_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	runs_dir.push("../../data/runs");

	for run_entry in fs::read_dir(runs_dir).unwrap() {
		let run_path = run_entry.unwrap().path();

		for file_entry in fs::read_dir(run_path).unwrap() {
			let file_path = file_entry.unwrap().path();

			if file_path.extension().unwrap().to_string_lossy()
				== "trees"
			{
				let file = fs::File::open(&file_path).unwrap();
				let reader = BufReader::new(file);

				for line in reader.lines() {
					let line = line.unwrap();
					TreeBuilder::from_newick(&line)
						.unwrap();
				}
			}
		}
	}
}
