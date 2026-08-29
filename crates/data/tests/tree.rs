use anyhow::{Result, ensure};
use arbitrary::Unstructured;
use arbtest::arbtest;
use picoarrow::array::{ArrayUtf8, Nullable};

use std::collections::BTreeSet;

use data::{
	newick::{Edge, Node as NewickNode, Tree as NewickTree},
	tree::{BinaryRootedTree, Internal, Node},
};

fn nullable_values<'a>(
	values: impl IntoIterator<Item = Option<&'a str>>,
) -> ArrayUtf8<Nullable> {
	let mut out = ArrayUtf8::<Nullable>::new();
	for value in values {
		out.push(value).unwrap();
	}
	out
}

fn names(values: &[String]) -> ArrayUtf8<Nullable> {
	nullable_values(
		values.iter().map(|value| {
			(!value.is_empty()).then_some(value.as_str())
		}),
	)
}

fn str_names(values: &[&str]) -> ArrayUtf8<Nullable> {
	nullable_values(
		values.iter()
			.map(|value| (!value.is_empty()).then_some(*value)),
	)
}

fn nulls(len: usize) -> ArrayUtf8<Nullable> {
	nullable_values((0..len).map(|_| None))
}

fn tree(
	num_leaves: u32,
	children: Vec<u32>,
	edge_lengths: Vec<f64>,
	node_names: ArrayUtf8<Nullable>,
) -> Result<BinaryRootedTree> {
	let num_nodes = num_leaves as usize * 2 - 1;
	BinaryRootedTree::new(
		num_leaves,
		children.into_boxed_slice(),
		edge_lengths.into_boxed_slice(),
		node_names,
		nulls(num_nodes),
		nulls(num_nodes - 1),
	)
}

fn node(tree: &BinaryRootedTree, index: u32) -> Node {
	tree.nodes().nth(index as usize).unwrap()
}

fn internal(tree: &BinaryRootedTree, index: u32) -> Internal {
	tree.as_internal(node(tree, index)).unwrap()
}

fn indices(nodes: impl Iterator<Item = Node>) -> Vec<u32> {
	nodes.map(Node::index).collect()
}

fn indexed_tree(source: &str) -> Result<BinaryRootedTree> {
	let source = BinaryRootedTree::try_from(&NewickTree::parse(source)?)?;
	let num_leaves = source.num_leaves();
	let num_nodes = source.num_nodes();
	let mut mapping = vec![0; num_nodes as usize];
	let mut seen = vec![false; num_leaves as usize];
	let mut node_names = vec![String::new(); num_nodes as usize];

	for leaf in source.leaves() {
		let old_node = Node::from(leaf);
		let name = source.name(old_node).unwrap();
		let index = name.parse::<u32>()?;
		ensure!(index < num_leaves);
		ensure!(!seen[index as usize]);
		mapping[old_node.index() as usize] = index;
		seen[index as usize] = true;
		node_names[index as usize] = name.to_owned();
	}
	ensure!(seen.into_iter().all(|value| value));

	for internal in source.internals() {
		mapping[internal.index() as usize] = internal.index();
		if let Some(name) = source.name(internal.into()) {
			node_names[internal.index() as usize] = name.to_owned();
		}
	}

	let children = source
		.internals()
		.flat_map(|internal| {
			let (left, right) = source.children_of(internal);
			[
				mapping[left.index() as usize],
				mapping[right.index() as usize],
			]
		})
		.collect::<Vec<_>>();
	let mut edge_lengths = vec![0.0; source.num_edges() as usize];
	for child in source.edges() {
		edge_lengths[mapping[child.index() as usize] as usize] =
			source.edge_length(child).unwrap();
	}

	tree(num_leaves, children, edge_lengths, names(&node_names))
}

fn clades(tree: &BinaryRootedTree) -> BTreeSet<Vec<u32>> {
	let mut descendants = vec![Vec::new(); tree.num_nodes() as usize];
	let mut clades = BTreeSet::new();

	for node in tree.postorder() {
		if let Some(leaf) = tree.as_leaf(node) {
			descendants[node.index() as usize] = vec![leaf.index()];
			continue;
		}

		let internal = tree.as_internal(node).unwrap();
		let (left, right) = tree.children_of(internal);
		let mut leaves = descendants[left.index() as usize].clone();
		leaves.extend_from_slice(&descendants[right.index() as usize]);
		leaves.sort_unstable();
		if internal != tree.root() {
			clades.insert(leaves.clone());
		}
		descendants[node.index() as usize] = leaves;
	}

	clades
}

fn mrca_slow(tree: &BinaryRootedTree, first: Node, second: Node) -> Node {
	let mut ancestors = BTreeSet::new();
	let mut current = Some(first);
	while let Some(node) = current {
		ancestors.insert(node);
		current = tree.parent_of(node).map(Node::from);
	}

	let mut current = second;
	loop {
		if ancestors.contains(&current) {
			return current;
		}
		current = tree.parent_of(current).unwrap().into();
	}
}

fn lca_depths(tree: &BinaryRootedTree) -> Vec<u32> {
	let num_leaves = tree.num_leaves() as usize;
	let mut depths = vec![0; tree.num_nodes() as usize];
	for node in tree.preorder() {
		if let Some(internal) = tree.as_internal(node) {
			let (left, right) = tree.children_of(internal);
			depths[left.index() as usize] =
				depths[node.index() as usize] + 1;
			depths[right.index() as usize] =
				depths[node.index() as usize] + 1;
		}
	}

	let mut lca_depths = vec![0; num_leaves * num_leaves];
	for first in 0..num_leaves {
		for second in first + 1..num_leaves {
			let mut left = node(tree, first as u32);
			let mut right = node(tree, second as u32);
			while depths[left.index() as usize]
				> depths[right.index() as usize]
			{
				left = tree.parent_of(left).unwrap().into();
			}
			while depths[right.index() as usize]
				> depths[left.index() as usize]
			{
				right = tree.parent_of(right).unwrap().into();
			}
			while left != right {
				left = tree.parent_of(left).unwrap().into();
				right = tree.parent_of(right).unwrap().into();
			}
			let depth = depths[left.index() as usize];
			lca_depths[first * num_leaves + second] = depth;
			lca_depths[second * num_leaves + first] = depth;
		}
	}
	lca_depths
}

fn triplet_topology(
	lca_depths: &[u32],
	num_leaves: usize,
	triplet: [usize; 3],
) -> usize {
	let depth = |left: usize, right: usize| {
		lca_depths[left * num_leaves + right]
	};
	let [a, b, c] = triplet;
	let depths = [depth(a, b), depth(a, c), depth(b, c)];
	depths.iter()
		.position(|&value| value == *depths.iter().max().unwrap())
		.unwrap()
}

fn triplet_distance_slow(
	first: &BinaryRootedTree,
	second: &BinaryRootedTree,
) -> u128 {
	assert_eq!(first.num_leaves(), second.num_leaves());
	let num_leaves = first.num_leaves() as usize;
	let first_depths = lca_depths(first);
	let second_depths = lca_depths(second);
	let mut distance = 0;

	for a in 0..num_leaves {
		for b in a + 1..num_leaves {
			for c in b + 1..num_leaves {
				let triplet = [a, b, c];
				distance += u128::from(
					triplet_topology(
						&first_depths,
						num_leaves,
						triplet,
					) != triplet_topology(
						&second_depths,
						num_leaves,
						triplet,
					),
				);
			}
		}
	}

	distance
}

fn arbitrary_tree(
	u: &mut Unstructured<'_>,
	num_leaves: u32,
) -> arbitrary::Result<BinaryRootedTree> {
	let num_nodes = num_leaves * 2 - 1;
	let mut available = (0..num_leaves).collect::<Vec<_>>();
	let mut children = Vec::with_capacity((num_nodes - 1) as usize);

	for parent in num_leaves..num_nodes {
		let left_index = u.int_in_range(0..=available.len() - 1)?;
		let left = available.swap_remove(left_index);
		let right_index = u.int_in_range(0..=available.len() - 1)?;
		let right = available.swap_remove(right_index);
		children.extend([left, right]);
		available.push(parent);
	}

	let node_names = (0..num_nodes)
		.map(|node| {
			if node < num_leaves {
				format!("leaf_{node}")
			} else {
				String::new()
			}
		})
		.collect::<Vec<_>>();

	Ok(tree(
		num_leaves,
		children,
		vec![0.0; (num_nodes - 1) as usize],
		names(&node_names),
	)
	.unwrap())
}

#[test]
fn two_leaf_tree() -> Result<()> {
	let tree = tree(
		2,
		vec![0, 1],
		vec![0.1, 0.2],
		str_names(&["A", "B", "root"]),
	)?;

	assert_eq!(tree.num_nodes(), 3);
	assert_eq!(tree.num_leaves(), 2);
	assert_eq!(tree.num_internals(), 1);
	assert_eq!(tree.num_edges(), 2);
	assert_eq!(tree.root().index(), 2);
	assert_eq!(
		tree.nodes().map(Node::index).collect::<Vec<_>>(),
		vec![0, 1, 2]
	);
	assert_eq!(
		tree.leaves().map(|leaf| leaf.index()).collect::<Vec<_>>(),
		vec![0, 1]
	);
	assert_eq!(
		tree.internals()
			.map(|internal| internal.index())
			.collect::<Vec<_>>(),
		vec![2]
	);
	assert_eq!(
		tree.edges().map(Node::index).collect::<Vec<_>>(),
		vec![0, 1]
	);
	assert!(tree.is_leaf(node(&tree, 0)));
	assert!(!tree.is_leaf(node(&tree, 2)));
	assert!(tree.is_internal(node(&tree, 2)));
	assert!(!tree.is_internal(node(&tree, 0)));
	assert_eq!(tree.as_leaf(node(&tree, 0)).unwrap().index(), 0);
	assert_eq!(tree.as_leaf(node(&tree, 2)), None);
	assert_eq!(tree.as_internal(node(&tree, 2)), Some(tree.root()));
	assert_eq!(tree.as_internal(node(&tree, 0)), None);
	assert_eq!(
		tree.children_of(tree.root()),
		(node(&tree, 0), node(&tree, 1))
	);
	assert_eq!(tree.parent_of(node(&tree, 0)), Some(tree.root()));
	assert_eq!(tree.parent_of(node(&tree, 1)), Some(tree.root()));
	assert_eq!(tree.parent_of(node(&tree, 2)), None);
	assert_eq!(tree.edge_length(node(&tree, 0)), Some(0.1));
	assert_eq!(tree.edge_length(node(&tree, 1)), Some(0.2));
	assert_eq!(tree.edge_length(node(&tree, 2)), None);
	assert_eq!(tree.name(node(&tree, 0)), Some("A"));
	assert_eq!(tree.name(node(&tree, 2)), Some("root"));
	assert_eq!(tree.node_metadata(node(&tree, 0)), None);
	assert_eq!(tree.edge_metadata(node(&tree, 0)), None);
	assert_eq!(tree.edge_metadata(node(&tree, 2)), None);
	assert_eq!(tree.leaf_by_name("B").unwrap().index(), 1);
	assert_eq!(indices(tree.preorder()), vec![2, 0, 1]);
	assert_eq!(indices(tree.postorder()), vec![0, 1, 2]);

	Ok(())
}

#[test]
fn balanced_and_ladder_traversals() -> Result<()> {
	let balanced = tree(
		4,
		vec![0, 1, 2, 3, 4, 5],
		vec![1.0; 6],
		str_names(&["A", "B", "C", "D", "", "", ""]),
	)?;
	assert_eq!(indices(balanced.preorder()), vec![6, 4, 0, 1, 5, 2, 3]);
	assert_eq!(indices(balanced.postorder()), vec![0, 1, 4, 2, 3, 5, 6]);
	assert_eq!(balanced.name(node(&balanced, 4)), None);

	let ladder = tree(
		4,
		vec![0, 1, 2, 4, 3, 5],
		vec![1.0; 6],
		str_names(&["A", "B", "C", "D", "", "", ""]),
	)?;
	assert_eq!(indices(ladder.preorder()), vec![6, 3, 5, 2, 4, 0, 1]);
	assert_eq!(indices(ladder.postorder()), vec![3, 2, 0, 1, 4, 5, 6]);
	assert_eq!(
		ladder.parent_of(node(&ladder, 4)),
		Some(internal(&ladder, 5))
	);

	Ok(())
}

#[test]
fn preserves_child_order_and_edge_values() -> Result<()> {
	let tree = tree(
		3,
		vec![1, 0, 2, 3],
		vec![f64::NAN, f64::INFINITY, -1.5, 0.0],
		str_names(&["A", "B", "C", "AB", "root"]),
	)?;
	let newick = NewickTree::from(&tree);
	let roundtrip = BinaryRootedTree::try_from(&newick)?;

	let preorder_names = roundtrip
		.preorder()
		.map(|node| roundtrip.name(node).unwrap())
		.collect::<Vec<_>>();
	assert_eq!(preorder_names, vec!["root", "C", "AB", "B", "A"]);
	let a = Node::from(roundtrip.leaf_by_name("A").unwrap());
	let b = Node::from(roundtrip.leaf_by_name("B").unwrap());
	let c = Node::from(roundtrip.leaf_by_name("C").unwrap());
	assert!(roundtrip.edge_length(a).unwrap().is_nan());
	assert_eq!(roundtrip.edge_length(b), Some(f64::INFINITY));
	assert_eq!(roundtrip.edge_length(c), Some(-1.5));
	assert_eq!(roundtrip.edge_length(node(&roundtrip, 3)), Some(0.0));

	Ok(())
}

#[test]
fn newick_roundtrip() -> Result<()> {
	let source = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";
	let newick = NewickTree::parse(source)?;
	let tree = BinaryRootedTree::try_from(&newick)?;
	assert_eq!(NewickTree::from(&tree).into_string(), source);

	let mut named = NewickTree::new();
	let a = named.add_node(NewickNode::named("A"));
	let b = named.add_node(NewickNode::named("B"));
	let c = named.add_node(NewickNode::named("C"));
	let d = named.add_node(NewickNode::named("D"));
	let ab = named.add_node(NewickNode::named("AB"));
	let cd = named.add_node(NewickNode::named("CD"));
	let root = named.add_node(NewickNode::named("ROOT"));
	for (parent, child, length) in [
		(ab, a, 0.1),
		(ab, b, 0.2),
		(cd, c, 0.4),
		(cd, d, 0.5),
		(root, ab, 0.3),
		(root, cd, 0.6),
	] {
		named.add_edge(parent, child, Edge::from_distance(length));
	}
	named.set_root(root);
	let named_tree = BinaryRootedTree::try_from(&named)?;
	assert_eq!(named_tree.name(node(&named_tree, 4)), Some("AB"));
	assert_eq!(named_tree.name(node(&named_tree, 5)), Some("CD"));
	assert_eq!(named_tree.name(node(&named_tree, 6)), Some("ROOT"));
	assert_eq!(
		NewickTree::from(&named_tree).into_string(),
		"((A:0.1,B:0.2)AB:0.3,(C:0.4,D:0.5)CD:0.6)ROOT;"
	);

	Ok(())
}

#[test]
fn metadata_roundtrip() -> Result<()> {
	let mut newick = NewickTree::new();
	let a = newick.add_node(NewickNode::new(
		"A".to_owned(),
		"[&country=SE]".to_owned(),
	));
	let b = newick.add_node(NewickNode::named("B"));
	let root = newick.add_node(NewickNode::new(
		"ROOT".to_owned(),
		"[&source=hcv]".to_owned(),
	));
	newick.add_edge(
		root,
		a,
		Edge::new(Some(0.1), "[&rate=fast]".to_owned()),
	);
	newick.add_edge(root, b, Edge::from_distance(0.2));
	newick.set_root(root);

	let tree = BinaryRootedTree::try_from(&newick)?;
	let a = Node::from(tree.leaf_by_name("A").unwrap());
	assert_eq!(tree.node_metadata(a), Some("[&country=SE]"));
	assert_eq!(tree.edge_metadata(a), Some("[&rate=fast]"));
	assert_eq!(
		tree.node_metadata(tree.root().into()),
		Some("[&source=hcv]")
	);
	assert_eq!(tree.edge_metadata(tree.root().into()), None);

	let output = NewickTree::from(&tree);
	let output_a = output
		.leaves()
		.find(|&node| output.get_node(node).name() == "A")
		.unwrap();
	assert_eq!(output.get_node(output_a).attributes(), "[&country=SE]");
	assert_eq!(
		output.edge_to_parent(output_a).unwrap().attributes(),
		"[&rate=fast]"
	);
	assert_eq!(
		output.get_node(*output.root().unwrap()).attributes(),
		"[&source=hcv]"
	);

	Ok(())
}

#[test]
fn constructor_rejects_invalid_layouts() {
	assert!(BinaryRootedTree::new(
		1,
		Box::new([]),
		Box::new([]),
		str_names(&["A"]),
		nulls(1),
		nulls(0),
	)
	.is_err());

	for (children, lengths, labels) in [
		(vec![0], vec![1.0, 1.0], vec!["A", "B", ""]),
		(vec![0, 1], vec![1.0], vec!["A", "B", ""]),
		(vec![0, 1], vec![1.0, 1.0], vec!["A", "B"]),
	] {
		assert!(BinaryRootedTree::new(
			2,
			children.into_boxed_slice(),
			lengths.into_boxed_slice(),
			str_names(&labels),
			nulls(3),
			nulls(2),
		)
		.is_err());
	}

	assert!(BinaryRootedTree::new(
		2,
		vec![0, 1].into_boxed_slice(),
		vec![1.0, 1.0].into_boxed_slice(),
		str_names(&["A", "B", ""]),
		nulls(2),
		nulls(2),
	)
	.is_err());
	assert!(BinaryRootedTree::new(
		2,
		vec![0, 1].into_boxed_slice(),
		vec![1.0, 1.0].into_boxed_slice(),
		str_names(&["A", "B", ""]),
		nulls(3),
		nulls(1),
	)
	.is_err());

	for children in [vec![0, 3], vec![0, 2], vec![0, 0], vec![0, 3, 1, 2]] {
		let num_leaves = if children.len() == 2 { 2 } else { 3 };
		let num_nodes = num_leaves * 2 - 1;
		assert!(BinaryRootedTree::new(
			num_leaves,
			children.into_boxed_slice(),
			vec![1.0; num_nodes as usize - 1].into_boxed_slice(),
			str_names(&vec![""; num_nodes as usize]),
			nulls(num_nodes as usize),
			nulls(num_nodes as usize - 1),
		)
		.is_err());
	}
}

#[test]
fn newick_rejects_unsupported_structures() -> Result<()> {
	let mut rootless = NewickTree::new();
	rootless.add_node(NewickNode::named("A"));
	assert!(BinaryRootedTree::try_from(&rootless).is_err());
	rootless.set_root(10);
	assert!(BinaryRootedTree::try_from(&rootless).is_err());

	let mut single = NewickTree::new();
	let root = single.add_node(NewickNode::named("A"));
	single.set_root(root);
	assert!(BinaryRootedTree::try_from(&single).is_err());

	for source in ["(A:1,B:1,C:1);", "(A:1,B:);"] {
		let newick = NewickTree::parse(source)?;
		assert!(BinaryRootedTree::try_from(&newick).is_err());
	}

	let mut unary = NewickTree::new();
	let child = unary.add_node(NewickNode::named("A"));
	let root = unary.add_node(NewickNode::unnamed());
	unary.add_edge(root, child, Edge::from_distance(1.0));
	unary.set_root(root);
	assert!(BinaryRootedTree::try_from(&unary).is_err());

	let mut disconnected = NewickTree::parse("(A:1,B:1);")?;
	disconnected.add_node(NewickNode::named("C"));
	assert!(BinaryRootedTree::try_from(&disconnected).is_err());

	let mut invalid_edge = NewickTree::new();
	let root = invalid_edge.add_node(NewickNode::unnamed());
	invalid_edge.add_edge(root, 10, Edge::from_distance(1.0));
	invalid_edge.set_root(root);
	assert!(BinaryRootedTree::try_from(&invalid_edge).is_err());

	Ok(())
}

#[test]
fn newick_rejects_multiple_parents_and_cycles() -> Result<()> {
	let mut multiple_parents = NewickTree::new();
	let child = multiple_parents.add_node(NewickNode::named("A"));
	let other = multiple_parents.add_node(NewickNode::named("B"));
	let internal = multiple_parents.add_node(NewickNode::unnamed());
	let root = multiple_parents.add_node(NewickNode::unnamed());
	multiple_parents.add_edge(internal, child, Edge::from_distance(1.0));
	multiple_parents.add_edge(root, child, Edge::from_distance(1.0));
	multiple_parents.add_edge(root, other, Edge::from_distance(1.0));
	multiple_parents.set_root(root);
	assert!(BinaryRootedTree::try_from(&multiple_parents).is_err());

	let mut cycle = NewickTree::parse("(A:1,B:1);")?;
	let leaf_c = cycle.add_node(NewickNode::named("C"));
	let leaf_d = cycle.add_node(NewickNode::named("D"));
	let internal_c = cycle.add_node(NewickNode::unnamed());
	let internal_d = cycle.add_node(NewickNode::unnamed());
	cycle.add_edge(internal_c, internal_d, Edge::from_distance(1.0));
	cycle.add_edge(internal_c, leaf_c, Edge::from_distance(1.0));
	cycle.add_edge(internal_d, internal_c, Edge::from_distance(1.0));
	cycle.add_edge(internal_d, leaf_d, Edge::from_distance(1.0));
	assert!(BinaryRootedTree::try_from(&cycle).is_err());

	Ok(())
}

#[test]
fn deep_ladder_uses_iterative_traversal() -> Result<()> {
	const NUM_LEAVES: u32 = 10_000;
	let mut children = Vec::with_capacity((NUM_LEAVES as usize - 1) * 2);
	children.extend([0, 1]);
	for parent in NUM_LEAVES + 1..NUM_LEAVES * 2 - 1 {
		children.extend([parent - NUM_LEAVES + 1, parent - 1]);
	}

	let tree = tree(
		NUM_LEAVES,
		children,
		vec![1.0; (NUM_LEAVES as usize - 1) * 2],
		names(&vec![String::new(); NUM_LEAVES as usize * 2 - 1]),
	)?;
	let newick = NewickTree::from(&tree);
	let roundtrip = BinaryRootedTree::try_from(&newick)?;

	assert_eq!(tree.preorder().count(), tree.num_nodes() as usize);
	assert_eq!(tree.postorder().count(), tree.num_nodes() as usize);
	assert_eq!(roundtrip.num_nodes(), tree.num_nodes());
	assert_eq!(tree.triplet_distance(&tree), 0);
	assert_eq!(
		tree.mrca(node(&tree, 0), node(&tree, NUM_LEAVES - 1)),
		tree.root().into()
	);
	assert_eq!(
		tree.mrca(node(&tree, 0), node(&tree, 1)),
		node(&tree, NUM_LEAVES)
	);

	Ok(())
}

#[test]
fn random_binary_trees_roundtrip() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=40)?;
		let num_nodes = num_leaves * 2 - 1;
		let mut available: Vec<u32> = (0..num_leaves).collect();
		let mut children = Vec::with_capacity((num_nodes - 1) as usize);

		for parent in num_leaves..num_nodes {
			let left_index =
				u.int_in_range(0..=available.len() - 1)?;
			let left = available.swap_remove(left_index);
			let right_index =
				u.int_in_range(0..=available.len() - 1)?;
			let right = available.swap_remove(right_index);
			children.extend([left, right]);
			available.push(parent);
		}

		let lengths = (0..num_nodes - 1)
			.map(|_| {
				u.arbitrary::<u16>()
					.map(|value| f64::from(value) / 100.0)
			})
			.collect::<arbitrary::Result<Vec<_>>>()?;
		let labels = (0..num_nodes)
			.map(|node| {
				if node < num_leaves {
					format!("leaf_{node}")
				} else {
					String::new()
				}
			})
			.collect::<Vec<_>>();
		let tree = tree(num_leaves, children, lengths, names(&labels))
			.unwrap();

		for internal in tree.internals() {
			let (left, right) = tree.children_of(internal);
			assert_eq!(tree.parent_of(left), Some(internal));
			assert_eq!(tree.parent_of(right), Some(internal));
		}
		assert_eq!(tree.preorder().count(), num_nodes as usize);
		assert_eq!(tree.postorder().count(), num_nodes as usize);

		let first_newick = NewickTree::from(&tree);
		let roundtrip =
			BinaryRootedTree::try_from(&first_newick).unwrap();
		let second_newick = NewickTree::from(&roundtrip);
		assert_eq!(
			first_newick.into_string(),
			second_newick.into_string()
		);

		Ok(())
	});
}

#[test]
fn ordered_leaf_attachment() -> Result<()> {
	for (source, expected) in [
		("(((0:0,1:0):0,3:0):0,2:0);", vec![0, -1, -1]),
		("(((0:0,2:0):0,3:0):0,1:0);", vec![0, 0, -2]),
		("(((1:0,2:0):0,3:0):0,0:0);", vec![0, 1, -2]),
		(
			"(((0:0,(1:0,5:0):0):0,(3:0,4:0):0):0,2:0);",
			vec![0, -1, -1, 3, 1],
		),
		(
			"((0:0,1:0):0,(((5:0,3:0):0,4:0):0,2:0):0);",
			vec![0, -1, 2, 3, 3],
		),
		("((0:0,(2:0,3:0):0):0,(1:0,4:0):0);", vec![0, 0, 2, 1]),
		("((0:0,((1:0,3:0):0,4:0):0):0,2:0);", vec![0, -1, 1, -3]),
	] {
		assert_eq!(indexed_tree(source)?.ola(), expected);
	}

	Ok(())
}

#[test]
fn robinson_foulds() -> Result<()> {
	for (first, second, expected) in [
		(
			"((0:0,1:0):0,(2:0,3:0):0);",
			"((0:0,1:0):0,(2:0,3:0):0);",
			0,
		),
		(
			"((0:0,1:0):0,(2:0,3:0):0);",
			"(((0:0,1:0):0,2:0):0,3:0);",
			2,
		),
		(
			"((0:0,1:0):0,(2:0,3:0):0);",
			"((0:0,2:0):0,(1:0,3:0):0);",
			4,
		),
		(
			"((0:0,1:0):0,(2:0,3:0):0);",
			"((1:0,0:0):0,(3:0,2:0):0);",
			0,
		),
		(
			"((0:0,1:0):0,(2:0,(3:0,4:0):0):0);",
			"((0:0,1:0):0,(2:0,(3:0,4:0):0):0);",
			0,
		),
		(
			"((0:0,1:0):0,(2:0,(3:0,4:0):0):0);",
			"((0:0,1:0):0,((2:0,3:0):0,4:0):0);",
			2,
		),
		(
			"((0:0,1:0):0,(2:0,(3:0,4:0):0):0);",
			"((((0:0,1:0):0,2:0):0,3:0):0,4:0);",
			4,
		),
		(
			"(0:0,(1:0,(2:0,(3:0,4:0):0):0):0);",
			"((((0:0,1:0):0,2:0):0,3:0):0,4:0);",
			6,
		),
		(
			"(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);",
			"(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);",
			0,
		),
		(
			"(((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0);",
			"((0:0,1:0):0,((2:0,3:0):0,(4:0,5:0):0):0);",
			2,
		),
		(
			"((((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0):0,(6:0,7:0):0);",
			"((((0:0,1:0):0,(2:0,3:0):0):0,(4:0,5:0):0):0,(6:0,7:0):0);",
			0,
		),
	] {
		let first = indexed_tree(first)?;
		let second = indexed_tree(second)?;
		assert_eq!(first.robinson_foulds(&second), expected);
		assert_eq!(second.robinson_foulds(&first), expected);
	}

	Ok(())
}

#[test]
fn random_robinson_foulds() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=1_000)?;
		let first = arbitrary_tree(u, num_leaves)?;
		let second = arbitrary_tree(u, num_leaves)?;
		let first_clades = clades(&first);
		let second_clades = clades(&second);
		let expected = first_clades
			.symmetric_difference(&second_clades)
			.count() as u32;

		assert_eq!(first.robinson_foulds(&second), expected);
		assert_eq!(second.robinson_foulds(&first), expected);

		Ok(())
	});
}

#[test]
fn mrca() -> Result<()> {
	let balanced = indexed_tree("((0:0,1:0):0,(2:0,3:0):0);")?;
	let zero = node(&balanced, 0);
	let one = node(&balanced, 1);
	let two = node(&balanced, 2);
	let left = node(&balanced, 4);
	let right = node(&balanced, 5);
	let root = Node::from(balanced.root());

	for (first, second, expected) in [
		(zero, zero, zero),
		(left, left, left),
		(zero, one, left),
		(one, zero, left),
		(zero, left, left),
		(left, zero, left),
		(two, right, right),
		(zero, two, root),
		(root, zero, root),
	] {
		assert_eq!(balanced.mrca(first, second), expected);
	}

	let ladder = indexed_tree("(0:0,(1:0,(2:0,(3:0,4:0):0):0):0);")?;
	for first in ladder.nodes() {
		for second in ladder.nodes() {
			assert_eq!(
				ladder.mrca(first, second),
				mrca_slow(&ladder, first, second)
			);
		}
	}

	Ok(())
}

#[test]
fn random_mrca() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=1_000)?;
		let tree = arbitrary_tree(u, num_leaves)?;
		let first =
			node(&tree, u.int_in_range(0..=tree.num_nodes() - 1)?);
		let second =
			node(&tree, u.int_in_range(0..=tree.num_nodes() - 1)?);
		let expected = mrca_slow(&tree, first, second);

		assert_eq!(tree.mrca(first, second), expected);
		assert_eq!(tree.mrca(second, first), expected);

		Ok(())
	});
}

#[test]
fn triplet_distance() -> Result<()> {
	let trees = [
		indexed_tree("((0:0,1:0):0,2:0);")?,
		indexed_tree("((0:0,2:0):0,1:0);")?,
		indexed_tree("((1:0,2:0):0,0:0);")?,
	];

	for (index, first) in trees.iter().enumerate() {
		assert_eq!(first.triplet_distance(first), 0);
		for second in &trees[index + 1..] {
			assert_eq!(first.triplet_distance(second), 1);
			assert_eq!(second.triplet_distance(first), 1);
		}
	}

	let first = indexed_tree("((0:0,(1:0,2:0):0):0,(3:0,4:0):0);")?;
	let second = indexed_tree("((1:0,(0:0,2:0):0):0,(3:0,4:0):0);")?;
	assert_eq!(first.triplet_distance(&second), 1);
	assert_eq!(second.triplet_distance(&first), 1);
	assert_eq!(triplet_distance_slow(&first, &second), 1);

	let first = indexed_tree("((0:0,1:0):0,(2:0,3:0):0);")?;
	let second = indexed_tree("(0:0,(1:0,(2:0,3:0):0):0);")?;
	assert_eq!(
		first.triplet_distance(&second),
		triplet_distance_slow(&first, &second)
	);
	assert_ne!(first.triplet_distance(&second), 0);

	Ok(())
}

#[test]
fn random_triplet_distance() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=24)?;
		let first = arbitrary_tree(u, num_leaves)?;
		let second = arbitrary_tree(u, num_leaves)?;
		let expected = triplet_distance_slow(&first, &second);

		assert_eq!(first.triplet_distance(&second), expected);
		assert_eq!(second.triplet_distance(&first), expected);

		Ok(())
	});
}

#[test]
fn random_large_triplet_distance_properties() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=1_000)?;
		let first = arbitrary_tree(u, num_leaves)?;
		let second = arbitrary_tree(u, num_leaves)?;
		let distance = first.triplet_distance(&second);

		assert_eq!(first.triplet_distance(&first), 0);
		assert_eq!(second.triplet_distance(&first), distance);
		assert!(distance <= choose3_for_test(num_leaves));

		Ok(())
	});
}

fn choose3_for_test(value: u32) -> u128 {
	let value = u128::from(value);
	value * value.saturating_sub(1) * value.saturating_sub(2) / 6
}
