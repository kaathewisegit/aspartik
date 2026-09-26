use anyhow::{Result, ensure};
use arbitrary::Unstructured;
use arbtest::arbtest;
use buffer::Buffer;
use computare_core::assert_almost_eq;
use picoarrow::array::{ArrayUtf8, Nullable};
use rand::SeedableRng;
use rand_pcg::Pcg64;

use std::collections::{BTreeMap, BTreeSet};

use data::tree::{
	BinaryTree, Internal, Node, SvgOptions, branch_score,
	branch_score_matrix, parse_newick, robinson_foulds_matrix,
	triplet_distance_matrix,
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
) -> Result<BinaryTree> {
	tree_with_root(
		num_leaves,
		num_leaves * 2 - 2,
		children,
		edge_lengths,
		node_names,
	)
}

fn tree_with_root(
	num_leaves: u32,
	root: u32,
	children: Vec<u32>,
	edge_lengths: Vec<f64>,
	node_names: ArrayUtf8<Nullable>,
) -> Result<BinaryTree> {
	let num_nodes = num_leaves as usize * 2 - 1;
	let mut parents = vec![u32::MAX; num_nodes];
	for (offset, pair) in children.as_chunks::<2>().0.iter().enumerate() {
		for &child in pair {
			parents[child as usize] = num_leaves + offset as u32;
		}
	}
	BinaryTree::new(
		num_leaves,
		root,
		Buffer::from_slice(&children),
		Buffer::from_slice(&parents),
		Buffer::from_slice(&edge_lengths),
		node_names,
		nulls(num_nodes),
		nulls(num_nodes - 1),
	)
}

fn node(tree: &BinaryTree, index: u32) -> Node {
	tree.nodes().nth(index as usize).unwrap()
}

fn topology(tree: &BinaryTree) -> Vec<[u32; 2]> {
	tree.internals()
		.map(|internal| tree.children_of(internal).map(Node::u32))
		.collect()
}

fn assert_random_tree(tree: &BinaryTree, num_leaves: u32) {
	assert_eq!(tree.num_leaves(), num_leaves);
	assert_eq!(tree.num_nodes(), num_leaves * 2 - 1);
	assert_eq!(tree.num_edges(), num_leaves * 2 - 2);
	assert_eq!(tree.root().u32(), num_leaves * 2 - 2);
	assert_eq!(tree.preorder().count(), tree.num_nodes() as usize);
	assert_eq!(tree.postorder().count(), tree.num_nodes() as usize);

	let mut seen = vec![false; tree.num_nodes() as usize];
	for node in tree.preorder() {
		assert!(!seen[node.usize()]);
		seen[node.usize()] = true;
		assert_eq!(tree.name(node), None);
		assert_eq!(tree.node_metadata(node), None);

		if let Some(internal) = tree.as_internal(node) {
			for child in tree.children_of(internal) {
				assert_eq!(
					tree.parent_of(child),
					Some(internal)
				);
			}
		}

		if node == tree.root().into() {
			assert_eq!(tree.parent_of(node), None);
			assert_eq!(tree.edge_length(node), None);
		} else {
			assert_eq!(tree.edge_length(node), Some(1.0));
		}
		assert_eq!(tree.edge_metadata(node), None);
	}
	assert!(seen.into_iter().all(|value| value));
}

fn internal(tree: &BinaryTree, index: u32) -> Internal {
	tree.as_internal(node(tree, index)).unwrap()
}

fn indices(nodes: impl Iterator<Item = Node>) -> Vec<u32> {
	nodes.map(Node::u32).collect()
}

fn indexed_tree(source: &str) -> Result<BinaryTree> {
	let source = parse_newick(source)?.into_binary()?;
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
		mapping[old_node.usize()] = index;
		seen[index as usize] = true;
		node_names[index as usize] = name.to_owned();
	}
	ensure!(seen.into_iter().all(|value| value));

	let internal_order = source
		.postorder()
		.filter_map(|node| source.as_internal(node))
		.collect::<Vec<_>>();
	for (offset, internal) in internal_order.into_iter().enumerate() {
		mapping[internal.usize()] = num_leaves + offset as u32;
		if let Some(name) = source.name(internal.into()) {
			node_names[(num_leaves + offset as u32) as usize] =
				name.to_owned();
		}
	}

	let mut children = vec![0; source.num_internals() as usize * 2];
	for internal in source.internals() {
		let mapped = mapping[internal.usize()];
		let offset = (mapped - num_leaves) as usize * 2;
		let [left, right] = source.children_of(internal);
		children[offset] = mapping[left.usize()];
		children[offset + 1] = mapping[right.usize()];
	}
	let root = mapping[source.root().usize()];
	let mut edge_lengths = vec![0.0; source.num_edges() as usize];
	for old_child in source.edges() {
		let child = mapping[old_child.usize()];
		let edge = child - u32::from(child > root);
		edge_lengths[edge as usize] =
			source.edge_length(old_child).unwrap();
	}

	tree_with_root(
		num_leaves,
		root,
		children,
		edge_lengths,
		names(&node_names),
	)
}

fn clades(tree: &BinaryTree) -> BTreeSet<Vec<u32>> {
	let mut descendants = vec![Vec::new(); tree.num_nodes() as usize];
	let mut clades = BTreeSet::new();

	for node in tree.postorder() {
		if let Some(leaf) = tree.as_leaf(node) {
			descendants[node.usize()] = vec![leaf.u32()];
			continue;
		}

		let internal = tree.as_internal(node).unwrap();
		let [left, right] = tree.children_of(internal);
		let mut leaves = descendants[left.usize()].clone();
		leaves.extend_from_slice(&descendants[right.usize()]);
		leaves.sort_unstable();
		if internal != tree.root() {
			clades.insert(leaves.clone());
		}
		descendants[node.usize()] = leaves;
	}

	clades
}

fn branch_clades(tree: &BinaryTree) -> BTreeMap<Vec<u32>, f64> {
	let mut descendants = vec![Vec::new(); tree.num_nodes() as usize];
	let mut clades = BTreeMap::new();
	for node in tree.postorder() {
		let leaves = if let Some(leaf) = tree.as_leaf(node) {
			vec![leaf.u32()]
		} else {
			let internal = tree.as_internal(node).unwrap();
			let [left, right] = tree.children_of(internal);
			let mut leaves = descendants[left.usize()].clone();
			leaves.extend_from_slice(&descendants[right.usize()]);
			leaves.sort_unstable();
			leaves
		};
		if node != tree.root().into() {
			clades.insert(
				leaves.clone(),
				tree.edge_length(node).unwrap(),
			);
		}
		descendants[node.usize()] = leaves;
	}
	clades
}

fn branch_score_slow(first: &BinaryTree, second: &BinaryTree) -> f64 {
	let mut first = branch_clades(first);
	let second = branch_clades(second);
	let mut squared = 0.0;
	for (clade, length) in second {
		let other = first.remove(&clade).unwrap_or(0.0);
		squared += (other - length).powi(2);
	}
	squared += first.values().map(|length| length.powi(2)).sum::<f64>();
	squared.sqrt()
}

fn mrca_slow(tree: &BinaryTree, first: Node, second: Node) -> Node {
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

fn lca_depths(tree: &BinaryTree) -> Vec<u32> {
	let num_leaves = tree.num_leaves() as usize;
	let mut depths = vec![0; tree.num_nodes() as usize];
	for node in tree.preorder() {
		if let Some(internal) = tree.as_internal(node) {
			let [left, right] = tree.children_of(internal);
			depths[left.usize()] = depths[node.usize()] + 1;
			depths[right.usize()] = depths[node.usize()] + 1;
		}
	}

	let mut lca_depths = vec![0; num_leaves * num_leaves];
	for first in 0..num_leaves {
		for second in first + 1..num_leaves {
			let mut left = node(tree, first as u32);
			let mut right = node(tree, second as u32);
			while depths[left.usize()] > depths[right.usize()] {
				left = tree.parent_of(left).unwrap().into();
			}
			while depths[right.usize()] > depths[left.usize()] {
				right = tree.parent_of(right).unwrap().into();
			}
			while left != right {
				left = tree.parent_of(left).unwrap().into();
				right = tree.parent_of(right).unwrap().into();
			}
			let depth = depths[left.usize()];
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

fn triplet_distance_slow(first: &BinaryTree, second: &BinaryTree) -> u128 {
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
) -> arbitrary::Result<BinaryTree> {
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

	let mut internal_ids = (num_leaves..num_nodes).collect::<Vec<_>>();
	for last in (1..internal_ids.len()).rev() {
		let index = u.int_in_range(0..=last)?;
		internal_ids.swap(last, index);
	}
	let mut mapping = (0..num_nodes).collect::<Vec<_>>();
	for (offset, id) in internal_ids.into_iter().enumerate() {
		mapping[num_leaves as usize + offset] = id;
	}
	let mut remapped_children = vec![0; children.len()];
	for old_parent in num_leaves..num_nodes {
		let old_offset = (old_parent - num_leaves) as usize * 2;
		let new_parent = mapping[old_parent as usize];
		let new_offset = (new_parent - num_leaves) as usize * 2;
		remapped_children[new_offset] =
			mapping[children[old_offset] as usize];
		remapped_children[new_offset + 1] =
			mapping[children[old_offset + 1] as usize];
	}
	let root = mapping[(num_nodes - 1) as usize];

	let node_names = (0..num_nodes)
		.map(|node| {
			if node < num_leaves {
				format!("leaf_{node}")
			} else {
				String::new()
			}
		})
		.collect::<Vec<_>>();

	Ok(tree_with_root(
		num_leaves,
		root,
		remapped_children,
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
	assert_eq!(tree.root().u32(), 2);
	assert_eq!(
		tree.nodes().map(Node::u32).collect::<Vec<_>>(),
		vec![0, 1, 2]
	);
	assert_eq!(
		tree.leaves().map(|leaf| leaf.u32()).collect::<Vec<_>>(),
		vec![0, 1]
	);
	assert_eq!(
		tree.internals()
			.map(|internal| internal.u32())
			.collect::<Vec<_>>(),
		vec![2]
	);
	assert_eq!(tree.edges().map(Node::u32).collect::<Vec<_>>(), vec![0, 1]);
	assert!(tree.is_leaf(node(&tree, 0)));
	assert!(!tree.is_leaf(node(&tree, 2)));
	assert!(tree.is_internal(node(&tree, 2)));
	assert!(!tree.is_internal(node(&tree, 0)));
	assert_eq!(tree.as_leaf(node(&tree, 0)).unwrap().u32(), 0);
	assert_eq!(tree.as_leaf(node(&tree, 2)), None);
	assert_eq!(tree.as_internal(node(&tree, 2)), Some(tree.root()));
	assert_eq!(tree.as_internal(node(&tree, 0)), None);
	assert_eq!(
		tree.children_of(tree.root()),
		[node(&tree, 0), node(&tree, 1)]
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
	assert_eq!(tree.leaf_by_name("B").unwrap().u32(), 1);
	assert_eq!(indices(tree.preorder()), vec![2, 0, 1]);
	assert_eq!(indices(tree.postorder()), vec![0, 1, 2]);

	Ok(())
}

#[test]
fn constructor_accepts_owned_buffers() -> Result<()> {
	let children = Buffer::from_slice(&[1, 0]);
	let edge_lengths = Buffer::from_slice(&[1.0, 2.0]);
	let tree = BinaryTree::new(
		2,
		2,
		children,
		Buffer::from_slice(&[2, 2, u32::MAX]),
		edge_lengths,
		str_names(&["A", "B", "root"]),
		nulls(3),
		nulls(2),
	)?;

	assert_eq!(tree.children_of(tree.root()).map(Node::u32), [1, 0]);
	assert_eq!(tree.edge_length(node(&tree, 0)), Some(1.0));
	assert_eq!(tree.edge_length(node(&tree, 1)), Some(2.0));
	tree.validate()?;
	Ok(())
}

#[test]
fn canonical_constructor_relabels_internals() -> Result<()> {
	let tree = BinaryTree::from_children(
		4,
		4,
		Buffer::from_slice(&[5, 6, 0, 1, 2, 3]),
		Buffer::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
		str_names(&["A", "B", "C", "D", "ROOT", "AB", "CD"]),
		nullable_values([
			None,
			None,
			None,
			None,
			Some("[&node=root]"),
			Some("[&node=AB]"),
			Some("[&node=CD]"),
		]),
		nullable_values([
			None,
			None,
			None,
			None,
			Some("[&edge=AB]"),
			Some("[&edge=CD]"),
		]),
	)?;

	tree.validate()?;
	assert_eq!(tree.root().u32(), 6);
	assert_eq!(
		tree.postorder()
			.filter(|&node| tree.is_internal(node))
			.map(Node::u32)
			.collect::<Vec<_>>(),
		vec![4, 5, 6]
	);
	assert_eq!(tree.name(node(&tree, 4)), Some("AB"));
	assert_eq!(tree.name(node(&tree, 5)), Some("CD"));
	assert_eq!(tree.name(node(&tree, 6)), Some("ROOT"));
	assert_eq!(tree.node_metadata(node(&tree, 4)), Some("[&node=AB]"));
	assert_eq!(tree.node_metadata(node(&tree, 6)), Some("[&node=root]"));
	assert_eq!(tree.edge_metadata(node(&tree, 4)), Some("[&edge=AB]"));
	assert_eq!(tree.edge_metadata(node(&tree, 5)), Some("[&edge=CD]"));
	assert_eq!(tree.edge_length(node(&tree, 4)), Some(0.5));
	assert_eq!(tree.edge_length(node(&tree, 5)), Some(0.6));
	assert_eq!(
		tree.to_newick()?,
		"((A:0.1,B:0.2)AB[&node=AB]:0.5[&edge=AB],(C:0.3,D:0.4)CD[&node=CD]:0.6[&edge=CD])ROOT[&node=root];"
	);
	Ok(())
}

#[test]
fn canonical_names_and_topology() -> Result<()> {
	let first = parse_newick(
		"((B:2[&edge=B],A[&node=A]:1)AB[&node=AB]:5[&edge=AB],(D:4,C:3)CD:6)ROOT;",
	)?
	.into_binary()?
	.canonical()?;
	let second = parse_newick(
		"((C:3,D:4)CD:6,(A[&node=A]:1,B:2[&edge=B])AB[&node=AB]:5[&edge=AB])ROOT;",
	)?
	.into_binary()?
	.canonical()?;

	assert_eq!(first.to_newick()?, second.to_newick()?);
	assert_eq!(first.to_newick()?, first.canonical()?.to_newick()?);
	assert_eq!(first.root().u32(), 6);
	assert_eq!(
		first.leaves()
			.map(|leaf| first.name(leaf.into()).unwrap())
			.collect::<Vec<_>>(),
		["A", "B", "C", "D"]
	);
	assert_eq!(
		first.edge_length(first.leaf_by_name("B").unwrap().into()),
		Some(2.0)
	);
	assert_eq!(
		first.edge_metadata(first.leaf_by_name("B").unwrap().into()),
		Some("[&edge=B]")
	);
	assert_eq!(
		first.node_metadata(first.leaf_by_name("A").unwrap().into()),
		Some("[&node=A]")
	);
	assert_eq!(
		first.postorder()
			.filter(|&node| first.is_internal(node))
			.map(Node::u32)
			.collect::<Vec<_>>(),
		vec![4, 5, 6]
	);
	Ok(())
}

#[test]
fn canonical_rejects_missing_and_duplicate_names() -> Result<()> {
	let unnamed = BinaryTree::random(3, &mut Pcg64::seed_from_u64(4))?;
	assert!(unnamed.canonical().is_err());
	for labels in [["A", "", "root"], ["A", "A", "root"]] {
		let tree = tree(
			2,
			vec![0, 1],
			vec![1.0, 2.0],
			str_names(&labels),
		)?;
		assert!(tree.canonical().is_err());
	}
	Ok(())
}

#[test]
fn canonical_is_independent_of_child_order_and_internal_ids() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=80)?;
		let original = arbitrary_tree(u, num_leaves)?;
		let mut children = topology(&original)
			.into_iter()
			.flatten()
			.collect::<Vec<_>>();
		for pair in children.as_chunks_mut::<2>().0 {
			pair.swap(0, 1);
		}
		let lengths = original
			.edges()
			.map(|child| original.edge_length(child).unwrap())
			.collect::<Vec<_>>();
		let labels = original
			.nodes()
			.map(|node| {
				original.name(node)
					.unwrap_or_default()
					.to_owned()
			})
			.collect::<Vec<_>>();
		let swapped = tree_with_root(
			num_leaves,
			original.root().u32(),
			children,
			lengths,
			names(&labels),
		)
		.unwrap();
		let first = original.canonical().unwrap();
		let second = swapped.canonical().unwrap();
		let reparsed = parse_newick(&original.to_newick().unwrap())
			.unwrap()
			.into_binary()
			.unwrap()
			.canonical()
			.unwrap();
		assert_eq!(
			first.to_newick().unwrap(),
			second.to_newick().unwrap()
		);
		assert_eq!(
			first.to_newick().unwrap(),
			reparsed.to_newick().unwrap()
		);
		assert_eq!(
			first.to_newick().unwrap(),
			first.canonical().unwrap().to_newick().unwrap()
		);
		for internal in first.internals() {
			let [left, right] = first.children_of(internal);
			assert!(left.u32() < right.u32());
		}
		first.validate().unwrap();
		Ok(())
	});
}

#[test]
fn canonical_deep_ladder() -> Result<()> {
	const NUM_LEAVES: u32 = 10_000;
	let mut children = Vec::with_capacity((NUM_LEAVES as usize - 1) * 2);
	children.extend([0, 1]);
	for parent in NUM_LEAVES + 1..NUM_LEAVES * 2 - 1 {
		children.extend([parent - NUM_LEAVES + 1, parent - 1]);
	}
	let labels = (0..NUM_LEAVES)
		.map(|leaf| format!("leaf_{:05}", NUM_LEAVES - leaf))
		.chain((NUM_LEAVES..NUM_LEAVES * 2 - 1).map(|_| String::new()))
		.collect::<Vec<_>>();
	let tree = tree(
		NUM_LEAVES,
		children,
		vec![1.0; (NUM_LEAVES as usize - 1) * 2],
		names(&labels),
	)?;
	let canonical = tree.canonical()?;
	assert_eq!(canonical.num_nodes(), tree.num_nodes());
	assert_eq!(
		canonical.name(canonical.leaves().next().unwrap().into()),
		Some("leaf_00001")
	);
	assert_eq!(canonical.root().u32(), canonical.num_nodes() - 1);
	assert_eq!(canonical.robinson_foulds(&canonical), 0);
	canonical.validate()?;
	Ok(())
}

#[test]
fn random_canonical_constructor() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=100)?;
		let original = arbitrary_tree(u, num_leaves)?;
		let children = topology(&original)
			.into_iter()
			.flatten()
			.collect::<Vec<_>>();
		let lengths = original
			.edges()
			.map(|child| original.edge_length(child).unwrap())
			.collect::<Vec<_>>();
		let labels = original
			.nodes()
			.map(|node| {
				original.name(node)
					.unwrap_or_default()
					.to_owned()
			})
			.collect::<Vec<_>>();
		let canonical = BinaryTree::from_children(
			num_leaves,
			original.root().u32(),
			Buffer::from_slice(&children),
			Buffer::from_slice(&lengths),
			names(&labels),
			nulls(original.num_nodes() as usize),
			nulls(original.num_edges() as usize),
		)
		.unwrap();

		assert_eq!(
			canonical.to_newick().unwrap(),
			original.to_newick().unwrap()
		);
		assert_eq!(
			canonical
				.postorder()
				.filter(|&node| canonical.is_internal(node))
				.map(Node::u32)
				.collect::<Vec<_>>(),
			(num_leaves..canonical.num_nodes()).collect::<Vec<_>>()
		);
		canonical.validate().unwrap();
		Ok(())
	});
}

#[test]
fn constructor_rejects_inconsistent_parents() {
	for parents in [[u32::MAX, 2, u32::MAX], [2, 2, 0], [2, 2, 2]] {
		assert!(BinaryTree::new(
			2,
			2,
			Buffer::from_slice(&[0, 1]),
			Buffer::from_slice(&parents),
			Buffer::from_slice(&[1.0, 2.0]),
			str_names(&["A", "B", "root"]),
			nulls(3),
			nulls(2),
		)
		.is_err());
	}
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
fn explicit_nonterminal_root() -> Result<()> {
	let tree = BinaryTree::new(
		4,
		4,
		Buffer::from_slice(&[5, 6, 0, 1, 2, 3]),
		Buffer::from_slice(&[5, 5, 6, 6, u32::MAX, 4, 4]),
		Buffer::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
		str_names(&["A", "B", "C", "D", "ROOT", "AB", "CD"]),
		nulls(7),
		nulls(6),
	)?;

	assert_eq!(tree.root().u32(), 4);
	assert_eq!(
		tree.children_of(tree.root()),
		[node(&tree, 5), node(&tree, 6)]
	);
	assert_eq!(tree.parent_of(node(&tree, 5)), Some(tree.root()));
	assert_eq!(indices(tree.edges()), vec![0, 1, 2, 3, 5, 6]);
	assert_eq!(tree.edge_length(node(&tree, 4)), None);
	assert_eq!(tree.edge_length(node(&tree, 5)), Some(0.5));
	assert_eq!(tree.edge_length(node(&tree, 6)), Some(0.6));
	assert_eq!(indices(tree.preorder()), vec![4, 5, 0, 1, 6, 2, 3]);
	assert_eq!(indices(tree.postorder()), vec![0, 1, 5, 2, 3, 6, 4]);
	assert_eq!(tree.mrca(node(&tree, 0), node(&tree, 3)), node(&tree, 4));

	assert_eq!(
		tree.to_newick()?,
		"((A:0.1,B:0.2)AB:0.5,(C:0.3,D:0.4)CD:0.6)ROOT;"
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
	let roundtrip = parse_newick(&tree.to_newick()?)?.into_binary()?;

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
	let ab = roundtrip
		.nodes()
		.find(|&node| roundtrip.name(node) == Some("AB"))
		.unwrap();
	assert_eq!(roundtrip.edge_length(ab), Some(0.0));

	Ok(())
}

#[test]
fn newick_roundtrip() -> Result<()> {
	let source = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";
	let tree = parse_newick(source)?.into_binary()?;
	assert_eq!(tree.to_newick()?, source);

	let named_source = "((A:0.1,B:0.2)AB:0.3,(C:0.4,D:0.5)CD:0.6)ROOT;";
	let named_tree = parse_newick(named_source)?.into_binary()?;
	assert!(named_tree
		.nodes()
		.any(|node| named_tree.name(node) == Some("AB")));
	assert!(named_tree
		.nodes()
		.any(|node| named_tree.name(node) == Some("CD")));
	assert_eq!(named_tree.name(named_tree.root().into()), Some("ROOT"));
	assert_eq!(named_tree.to_newick()?, named_source);

	Ok(())
}

#[test]
fn metadata_roundtrip() -> Result<()> {
	let source = "(A[&country=SE]:0.1[&rate=fast],B:0.2)ROOT[&source=hcv];";
	let tree = parse_newick(source)?.into_binary()?;
	let a = Node::from(tree.leaf_by_name("A").unwrap());
	assert_eq!(tree.node_metadata(a), Some("[&country=SE]"));
	assert_eq!(tree.edge_metadata(a), Some("[&rate=fast]"));
	assert_eq!(
		tree.node_metadata(tree.root().into()),
		Some("[&source=hcv]")
	);
	assert_eq!(tree.edge_metadata(tree.root().into()), None);

	assert_eq!(tree.to_newick()?, source);

	Ok(())
}

#[test]
fn constructor_rejects_invalid_layouts() {
	assert!(BinaryTree::from_children(
		1,
		0,
		Buffer::from_slice(&[]),
		Buffer::from_slice(&[]),
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
		assert!(BinaryTree::from_children(
			2,
			2,
			Buffer::from_slice(&children),
			Buffer::from_slice(&lengths),
			str_names(&labels),
			nulls(3),
			nulls(2),
		)
		.is_err());
	}

	assert!(BinaryTree::from_children(
		2,
		0,
		Buffer::from_slice(&[0, 1]),
		Buffer::from_slice(&[1.0, 1.0]),
		str_names(&["A", "B", ""]),
		nulls(3),
		nulls(2),
	)
	.is_err());
	assert!(BinaryTree::from_children(
		2,
		3,
		Buffer::from_slice(&[0, 1]),
		Buffer::from_slice(&[1.0, 1.0]),
		str_names(&["A", "B", ""]),
		nulls(3),
		nulls(2),
	)
	.is_err());
	assert!(BinaryTree::from_children(
		2,
		2,
		Buffer::from_slice(&[0, 1]),
		Buffer::from_slice(&[1.0, 1.0]),
		str_names(&["A", "B", ""]),
		nulls(2),
		nulls(2),
	)
	.is_err());
	assert!(BinaryTree::from_children(
		2,
		2,
		Buffer::from_slice(&[0, 1]),
		Buffer::from_slice(&[1.0, 1.0]),
		str_names(&["A", "B", ""]),
		nulls(3),
		nulls(1),
	)
	.is_err());

	for children in [vec![0, 3], vec![0, 2], vec![0, 0], vec![0, 3, 1, 2]] {
		let num_leaves = if children.len() == 2 { 2 } else { 3 };
		let num_nodes = num_leaves * 2 - 1;
		assert!(BinaryTree::from_children(
			num_leaves,
			num_nodes - 1,
			Buffer::from_slice(&children),
			Buffer::from_slice(&vec![1.0; num_nodes as usize - 1]),
			str_names(&vec![""; num_nodes as usize]),
			nulls(num_nodes as usize),
			nulls(num_nodes as usize - 1),
		)
		.is_err());
	}
}

#[test]
fn from_children_rejects_oversized_leaf_count() {
	assert!(BinaryTree::from_children(
		u32::MAX / 2 + 1,
		0,
		Buffer::from_slice(&[]),
		Buffer::from_slice(&[]),
		nulls(0),
		nulls(0),
		nulls(0),
	)
	.is_err());
}

#[test]
fn newick_rejects_unsupported_structures() -> Result<()> {
	for source in ["A;", "(A:1);", "(A:1,B:1,C:1);", "(A:1,B:);"] {
		assert!(parse_newick(source)?.into_binary().is_err());
	}

	Ok(())
}

#[test]
fn newick_rejects_multiple_parents_and_cycles() -> Result<()> {
	assert!(parse_newick("((A:1)X#H1:1,(X#H1:1,B:1):1);")?
		.into_binary()
		.is_err());
	assert!(parse_newick("((A:1)X#H1:1,(X#H1:1,B:1)X#H1:1);").is_err());
	assert!(parse_newick("((A:1)X#H:1,B:1);").is_err());

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
	let roundtrip = parse_newick(&tree.to_newick()?)?.into_binary()?;

	assert_eq!(tree.preorder().count(), tree.num_nodes() as usize);
	assert_eq!(tree.postorder().count(), tree.num_nodes() as usize);
	assert_eq!(roundtrip.num_nodes(), tree.num_nodes());
	assert_eq!(tree.robinson_foulds(&tree), 0);
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
fn random_binary_tree() -> Result<()> {
	let mut rng = Pcg64::seed_from_u64(0);
	for num_leaves in [2, 3, 10, 100, 1_000] {
		let tree = BinaryTree::random(num_leaves, &mut rng)?;
		assert_random_tree(&tree, num_leaves);
	}

	Ok(())
}

#[test]
fn random_binary_tree_is_deterministic() -> Result<()> {
	let mut first_rng = Pcg64::seed_from_u64(0);
	let mut second_rng = Pcg64::seed_from_u64(0);
	let first = BinaryTree::random(100, &mut first_rng)?;
	let second = BinaryTree::random(100, &mut second_rng)?;

	assert_eq!(topology(&first), topology(&second));

	Ok(())
}

#[test]
fn random_binary_tree_leaf_count_bounds() {
	let mut rng = Pcg64::seed_from_u64(0);
	assert!(BinaryTree::random(0, &mut rng).is_err());
	assert!(BinaryTree::random(1, &mut rng).is_err());
	assert!(BinaryTree::random(2, &mut rng).is_ok());
	assert!(BinaryTree::random(u32::MAX / 2 + 1, &mut rng).is_err());
	assert!(BinaryTree::random(u32::MAX, &mut rng).is_err());
}

#[test]
fn random_binary_tree_many_seeds_and_large() -> Result<()> {
	let mut topologies = BTreeSet::new();
	for seed in 0..128 {
		let mut rng = Pcg64::seed_from_u64(seed);
		for num_leaves in 2..=12 {
			let tree = BinaryTree::random(num_leaves, &mut rng)?;
			assert_random_tree(&tree, num_leaves);
		}
		let tree = BinaryTree::random(64, &mut rng)?;
		assert_random_tree(&tree, 64);
		topologies.insert(topology(&tree));
	}
	assert!(topologies.len() > 120);

	let mut rng = Pcg64::seed_from_u64(0);
	let tree = BinaryTree::random(20_000, &mut rng)?;
	assert_random_tree(&tree, 20_000);

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
			let [left, right] = tree.children_of(internal);
			assert_eq!(tree.parent_of(left), Some(internal));
			assert_eq!(tree.parent_of(right), Some(internal));
		}
		assert_eq!(tree.preorder().count(), num_nodes as usize);
		assert_eq!(tree.postorder().count(), num_nodes as usize);

		let first_newick = tree.to_newick().unwrap();
		let roundtrip = parse_newick(&first_newick)
			.unwrap()
			.into_binary()
			.unwrap();
		assert_eq!(first_newick, roundtrip.to_newick().unwrap());

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
fn branch_score_distances() -> Result<()> {
	let first = indexed_tree("((0:1,1:2):3,(2:4,3:5):6);")?;
	let changed_lengths = indexed_tree("((0:2,1:4):6,(2:8,3:10):12);")?;
	let changed_topology = indexed_tree("((0:1,2:4):3,(1:2,3:5):6);")?;
	let zero = indexed_tree("((0:0,1:0):0,(2:0,3:0):0);")?;

	for (left, right) in [
		(&first, &first),
		(&first, &changed_lengths),
		(&first, &changed_topology),
		(&first, &zero),
	] {
		let expected = branch_score_slow(left, right);
		assert_almost_eq!(branch_score(left, right)?, expected);
		assert_almost_eq!(branch_score(right, left)?, expected);
	}
	assert_almost_eq!(branch_score(&first, &first)?, 0.0);
	assert_almost_eq!(branch_score(&zero, &zero)?, 0.0);
	assert!(branch_score(&first, &indexed_tree("(0:1,1:1);")?).is_err());

	Ok(())
}

#[test]
fn branch_score_reference_distances() -> Result<()> {
	// Expected values were generated with ape 5.8.1: dist.topo(first, second, method = "score").
	let cases = [
		(
			"((((0:1,1:1):0.4,(2:1,3:1):0.5):0.6,((4:1,5:1):0.7,(6:1,7:1):0.8):0.9):1,8:0);",
			"(((0:1,(1:1,2:1):0.45):0.65,(3:1,((4:1,5:1):0.75,(6:1,7:1):0.85):0.95):0.55):1,8:0);",
			1.3057564857200596,
		),
		(
			"(((((0:1,1:1.1):0.2,(2:1.2,3:1.3):0.3):0.4,((4:1.4,5:1.5):0.5,(6:1.6,7:1.7):0.6):0.7):0.8,(8:1.8,9:1.9):0.9):1,10:0);",
			"((((0:1,(1:1.1,2:1.2):0.25):0.35,(3:1.3,(4:1.4,5:1.5):0.55):0.45):0.65,((6:1.6,7:1.7):0.75,(8:1.8,9:1.9):0.85):0.95):1,10:0);",
			1.7776388834631178,
		),
		(
			"(((((0:1,1:1):0.2,(2:1,3:1):0.3):0.4,((4:1,5:1):0.5,(6:1,7:1):0.6):0.7):0.8,((8:1,9:1):0.9,(10:1,11:1):1):1.1):1,12:0);",
			"((((0:1,(1:1,2:1):0.25):0.35,(3:1,(4:1,5:1):0.45):0.55):0.65,((6:1,7:1):0.75,(8:1,(9:1,(10:1,11:1):0.85):0.95):1.05):1.15):1,12:0);",
			2.327_015_255_644_019,
		),
	];

	for (first, second, expected) in cases {
		let first = indexed_tree(first)?;
		let second = indexed_tree(second)?;
		assert_almost_eq!(branch_score(&first, &second)?, expected);
		assert_almost_eq!(branch_score(&second, &first)?, expected);
	}

	Ok(())
}

#[test]
fn multi_tree_branch_score() -> Result<()> {
	assert!(branch_score_matrix(&[])?.is_empty());

	let trees = [
		indexed_tree("((0:1,1:2):3,(2:4,3:5):6);")?,
		indexed_tree("((0:2,1:4):6,(2:8,3:10):12);")?,
		indexed_tree("((0:1,2:4):3,(1:2,3:5):6);")?,
		indexed_tree("((0:0,1:0):0,(2:0,3:0):0);")?,
	];
	let references = trees.iter().collect::<Vec<_>>();
	let distances = branch_score_matrix(&references)?;
	for (first_index, first) in trees.iter().enumerate() {
		assert_eq!(distances[first_index][first_index], 0.0);
		for (second_index, second) in trees.iter().enumerate() {
			assert_almost_eq!(
				distances[first_index][second_index],
				branch_score_slow(first, second)
			);
			assert_eq!(
				distances[first_index][second_index],
				distances[second_index][first_index]
			);
		}
	}

	assert_eq!(branch_score_matrix(&references[..1])?, [[0.0]]);
	let repeated = [references[0], references[0]];
	assert_eq!(branch_score_matrix(&repeated)?, [[0.0, 0.0], [0.0, 0.0]]);

	let close = indexed_tree("(0:1.000000001,1:1);")?;
	let base = indexed_tree("(0:1,1:1);")?;
	let close_distance = branch_score_matrix(&[&base, &close])?[0][1];
	let expected = branch_score(&base, &close)?;
	assert_almost_eq!(close_distance, expected, absolute = 1e-15);
	assert!(close_distance > 0.0);

	let mismatched_count = [
		&indexed_tree("(0:1,1:1);")?,
		&indexed_tree("((0:1,1:1):1,2:1);")?,
	];
	assert!(branch_score_matrix(&mismatched_count).is_err());

	let mismatched_names = [
		&parse_newick("(A:1,B:1);")?.into_binary()?,
		&parse_newick("(A:1,C:1);")?.into_binary()?,
	];
	assert!(branch_score_matrix(&mismatched_names).is_err());
	Ok(())
}

#[test]
fn random_multi_tree_branch_score() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=30)?;
		let num_trees = u.int_in_range(0_usize..=8)?;
		let num_nodes = num_leaves * 2 - 1;
		let mut node_names = vec![String::new(); num_nodes as usize];
		for leaf in 0..num_leaves {
			node_names[leaf as usize] = format!("leaf_{leaf}");
		}
		let trees = (0..num_trees)
			.map(|_| {
				let source = arbitrary_tree(u, num_leaves)?;
				let lengths = (0..source.num_edges())
					.map(|_| {
						u.int_in_range(0_u32..=100).map(
							|value| {
								f64::from(value)
									/ 10.0
							},
						)
					})
					.collect::<arbitrary::Result<Vec<_>>>(
					)?;
				Ok(tree_with_root(
					num_leaves,
					source.root().u32(),
					topology(&source)
						.into_iter()
						.flatten()
						.collect(),
					lengths,
					names(&node_names),
				)
				.unwrap())
			})
			.collect::<arbitrary::Result<Vec<_>>>()?;
		let references = trees.iter().collect::<Vec<_>>();
		let distances = branch_score_matrix(&references).unwrap();
		assert_eq!(distances.len(), num_trees);
		for (first_index, first) in trees.iter().enumerate() {
			assert_eq!(distances[first_index].len(), num_trees);
			for (second_index, second) in trees.iter().enumerate() {
				let expected =
					branch_score(first, second).unwrap();
				let actual =
					distances[first_index][second_index];
				assert_almost_eq!(
					actual,
					expected,
					relative = 1e-9
				);
			}
		}
		Ok(())
	});
}

#[test]
fn multi_tree_robinson_foulds() -> Result<()> {
	assert!(robinson_foulds_matrix(&[])?.is_empty());

	let trees = [
		&indexed_tree("((0:0,1:0):0,(2:0,3:0):0);")?,
		&indexed_tree("(((0:0,1:0):0,2:0):0,3:0);")?,
		&indexed_tree("((0:0,2:0):0,(1:0,3:0):0);")?,
		&indexed_tree("((1:0,0:0):0,(3:0,2:0):0);")?,
	];
	let distances = robinson_foulds_matrix(&trees)?;
	assert_eq!(
		distances,
		[[0, 2, 4, 0], [2, 0, 4, 2], [4, 4, 0, 4], [0, 2, 4, 0],]
	);

	let single = [&indexed_tree("(0:0,1:0);")?];
	assert_eq!(robinson_foulds_matrix(&single)?, [[0]]);

	let mismatched = [
		&indexed_tree("(0:0,1:0);")?,
		&indexed_tree("((0:0,1:0):0,2:0);")?,
	];
	assert!(robinson_foulds_matrix(&mismatched).is_err());

	Ok(())
}

#[test]
fn random_multi_tree_robinson_foulds() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=100)?;
		let num_trees = u.int_in_range(0_usize..=12)?;
		let trees = (0..num_trees)
			.map(|_| arbitrary_tree(u, num_leaves))
			.collect::<arbitrary::Result<Vec<_>>>()?;
		let trees = trees.iter().collect::<Vec<_>>();
		let distances = robinson_foulds_matrix(&trees).unwrap();

		assert_eq!(distances.len(), num_trees);
		for (first_index, first) in trees.iter().enumerate() {
			assert_eq!(distances[first_index].len(), num_trees);
			for (second_index, second) in trees.iter().enumerate() {
				assert_eq!(
					distances[first_index][second_index],
					first.robinson_foulds(second)
				);
			}
		}

		Ok(())
	});
}

#[test]
fn many_tree_robinson_foulds() -> Result<()> {
	let sources = [
		"((0:0,1:0):0,(2:0,3:0):0);",
		"(((0:0,1:0):0,2:0):0,3:0);",
		"((0:0,2:0):0,(1:0,3:0):0);",
	];
	let trees = (0..256)
		.map(|index| indexed_tree(sources[index % sources.len()]))
		.collect::<Result<Vec<_>>>()?;
	let trees = trees.iter().collect::<Vec<_>>();
	let distances = robinson_foulds_matrix(&trees)?;

	for (first_index, first) in trees.iter().enumerate() {
		for (second_index, second) in trees.iter().enumerate() {
			assert_eq!(
				distances[first_index][second_index],
				first.robinson_foulds(second)
			);
		}
	}

	Ok(())
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

	let first = indexed_tree(
		"((((14:0,(13:0,10:0):0):0,2:0):0,((8:0,(0:0,3:0):0):0,(((((17:0,1:0):0,4:0):0,(19:0,(15:0,7:0):0):0):0,(6:0,(11:0,18:0):0):0):0,9:0):0):0):0,(12:0,(5:0,16:0):0):0);",
	)?;
	let second = indexed_tree(
		"(((17:0,19:0):0,(4:0,8:0):0):0,((6:0,(((16:0,(11:0,0:0):0):0,14:0):0,1:0):0):0,((18:0,(9:0,2:0):0):0,(((7:0,5:0):0,15:0):0,(((10:0,13:0):0,3:0):0,12:0):0):0):0):0);",
	)?;
	assert_eq!(first.triplet_distance(&second), 858);
	assert_eq!(second.triplet_distance(&first), 858);

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
fn multi_tree_triplet_distance() -> Result<()> {
	assert!(triplet_distance_matrix(&[])?.is_empty());

	let trees = [
		indexed_tree("((0:0,1:0):0,2:0);")?,
		indexed_tree("((0:0,2:0):0,1:0);")?,
		indexed_tree("((1:0,2:0):0,0:0);")?,
	];
	let references = trees.iter().collect::<Vec<_>>();
	assert_eq!(
		triplet_distance_matrix(&references)?,
		[[0, 1, 1], [1, 0, 1], [1, 1, 0]]
	);
	assert_eq!(triplet_distance_matrix(&references[..1])?, [[0]]);

	let mismatch = [
		&indexed_tree("(0:0,1:0);")?,
		&indexed_tree("((0:0,1:0):0,2:0);")?,
	];
	assert!(triplet_distance_matrix(&mismatch).is_err());

	Ok(())
}

#[test]
fn random_multi_tree_triplet_distance() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=32)?;
		let num_trees = u.int_in_range(0_usize..=8)?;
		let trees = (0..num_trees)
			.map(|_| arbitrary_tree(u, num_leaves))
			.collect::<arbitrary::Result<Vec<_>>>()?;
		let references = trees.iter().collect::<Vec<_>>();
		let distances = triplet_distance_matrix(&references).unwrap();

		assert_eq!(distances.len(), num_trees);
		for (first_index, first) in trees.iter().enumerate() {
			assert_eq!(distances[first_index].len(), num_trees);
			for (second_index, second) in trees.iter().enumerate() {
				assert_eq!(
					distances[first_index][second_index],
					first.triplet_distance(second)
				);
			}
		}
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

#[test]
fn rectangular_slanted_and_tidy_layouts() -> Result<()> {
	let tree = indexed_tree("((0:1,1:3):2,(2:2,3:4):1);")?;
	let rectangular = tree.rectangular_layout(1.0)?;
	let slanted = tree.slanted_layout(1.0)?;
	let tidy = tree.tidy_layout(1.0)?;

	assert_eq!(rectangular.width(), 5.0);
	assert_eq!(rectangular.height(), 3.0);
	assert_eq!(slanted.points(), rectangular.points());
	assert_eq!(slanted.width(), rectangular.width());
	assert_eq!(slanted.height(), rectangular.height());
	assert!(tidy.width() == rectangular.width());
	assert!(tidy.height() <= rectangular.height());
	for layout in [&rectangular, &slanted, &tidy] {
		assert_eq!(layout.points().len(), tree.num_nodes() as usize);
		assert!(layout
			.points()
			.iter()
			.all(|point| point.x.is_finite()
				&& point.y.is_finite()));
		assert_eq!(layout.point(tree.root().into()).unwrap().x, 0.0);
		for child in tree.edges() {
			let parent = tree.parent_of(child).unwrap();
			let child_point = layout.point(child).unwrap();
			let parent_point = layout.point(parent.into()).unwrap();
			assert!(child_point.x >= parent_point.x);
		}
		for internal in tree.internals() {
			let parent = layout.point(internal.into()).unwrap();
			let [left, right] = tree.children_of(internal);
			let left = layout.point(left).unwrap();
			let right = layout.point(right).unwrap();
			assert!((parent.y - (left.y + right.y) / 2.0).abs()
				< 1e-12);
		}
	}
	let nonlayered =
		indexed_tree("(((0:2,1:1):1,2:1):1,(3:1,(4:1,5:2):1):1);")?;
	assert!(nonlayered.tidy_layout(1.0)?.height()
		< nonlayered.rectangular_layout(1.0)?.height());

	let two = indexed_tree("(0:1,1:2);")?;
	let layout = two.rectangular_layout(2.0)?;
	assert_eq!(
		layout.point(node(&two, 0)).unwrap(),
		data::tree::Point { x: 1.0, y: 0.0 }
	);
	assert_eq!(
		layout.point(node(&two, 1)).unwrap(),
		data::tree::Point { x: 2.0, y: 2.0 }
	);
	assert_eq!(
		layout.point(two.root().into()).unwrap(),
		data::tree::Point { x: 0.0, y: 1.0 }
	);

	Ok(())
}

#[test]
fn layout_rejects_invalid_values() -> Result<()> {
	let valid = indexed_tree("(0:1,1:2);")?;
	assert!(valid.rectangular_layout(0.0).is_err());
	assert!(valid.slanted_layout(f64::INFINITY).is_err());
	assert!(valid.tidy_layout(f64::NAN).is_err());

	for length in [f64::NAN, f64::INFINITY, -1.0] {
		let invalid = tree(
			2,
			vec![0, 1],
			vec![length, 1.0],
			str_names(&["0", "1", ""]),
		)?;
		assert!(invalid.rectangular_layout(1.0).is_err());
		assert!(invalid.slanted_layout(1.0).is_err());
		assert!(invalid.tidy_layout(1.0).is_err());
	}

	Ok(())
}

#[test]
fn svg_rendering() -> Result<()> {
	let tree = BinaryTree::new(
		2,
		2,
		Buffer::from_slice(&[0, 1]),
		Buffer::from_slice(&[2, 2, u32::MAX]),
		Buffer::from_slice(&[1.0, 2.0]),
		str_names(&["A<&\"'", "B", "root"]),
		nullable_values([Some("node<&\"'"), None, Some("root data")]),
		nullable_values([Some("edge<&\"'"), None]),
	)?;
	let layout = tree.rectangular_layout(1.0)?;
	let options = SvgOptions {
		x_scale: 100.0 / 3.0,
		..SvgOptions::default()
	};
	let svg =
		tree.to_svg(&layout, options, |_| "#123\"456", |_| "#abcdef")?;

	assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
	assert!(svg.ends_with("</svg>"));
	assert_eq!(svg.matches("<path ").count(), tree.num_edges() as usize);
	assert_eq!(svg.matches("<circle ").count(), tree.num_nodes() as usize);
	assert!(svg.contains("A&lt;&amp;&quot;&apos;"));
	assert!(svg.contains("node&lt;&amp;&quot;&apos;"));
	assert!(svg.contains("edge&lt;&amp;&quot;&apos;"));
	assert!(svg.contains("fill=\"#123&quot;456\""));
	assert!(svg.contains("width=\"122.87\""));
	assert!(svg.contains("H 53.33\""));
	assert!(svg.contains("cx=\"86.67\""));
	assert!(svg
		.contains("<g font-size=\"12\" dominant-baseline=\"middle\">"));
	assert_eq!(svg.matches("dominant-baseline").count(), 1);
	assert_eq!(svg.matches("font-size").count(), 1);
	let unnamed = tree.to_svg(
		&layout,
		SvgOptions {
			show_names: false,
			..options
		},
		|_| "black",
		|_| "black",
	)?;
	assert!(!unnamed.contains("<text"));
	assert!(!unnamed.contains("A&lt;&amp;&quot;&apos;"));
	assert!(unnamed.contains("node&lt;&amp;&quot;&apos;"));
	assert!(unnamed.contains("width=\"106.67\""));
	let slanted = tree.to_svg(
		&tree.slanted_layout(1.0)?,
		SvgOptions::default(),
		|_| "black",
		|_| "black",
	)?;
	assert_eq!(
		slanted.matches("<line ").count(),
		tree.num_edges() as usize
	);
	assert_eq!(slanted.matches("<path ").count(), 0);

	let tidy = tree.to_svg(
		&tree.tidy_layout(1.0)?,
		SvgOptions::default(),
		|_| "black",
		|_| "black",
	)?;
	assert_eq!(tidy.matches("<path ").count(), tree.num_edges() as usize);
	assert_eq!(tidy.matches("<line ").count(), 0);

	let invalid = SvgOptions {
		x_scale: 0.0,
		..SvgOptions::default()
	};
	assert!(tree
		.to_svg(&layout, invalid, |_| "black", |_| "black")
		.is_err());

	Ok(())
}

#[test]
fn random_layouts_are_finite() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2_u32..=1_000)?;
		let tree = arbitrary_tree(u, num_leaves)?;
		for layout in [
			tree.rectangular_layout(1.0).unwrap(),
			tree.slanted_layout(1.0).unwrap(),
			tree.tidy_layout(1.0).unwrap(),
		] {
			assert!(layout.width().is_finite());
			assert!(layout.height().is_finite());
			assert!(layout
				.points()
				.iter()
				.all(|point| point.x.is_finite()
					&& point.y.is_finite()));
		}
		Ok(())
	});
}

#[test]
fn large_ladder_layout() -> Result<()> {
	let num_leaves = 20_000_u32;
	let num_nodes = num_leaves * 2 - 1;
	let mut children = Vec::with_capacity((num_nodes - 1) as usize);
	children.extend([0, 1]);
	for offset in 1..num_leaves - 1 {
		children.extend([num_leaves + offset - 1, offset + 1]);
	}
	let labels = (0..num_nodes)
		.map(|node| {
			if node < num_leaves {
				node.to_string()
			} else {
				String::new()
			}
		})
		.collect::<Vec<_>>();
	let tree = tree(
		num_leaves,
		children,
		vec![1.0; (num_nodes - 1) as usize],
		names(&labels),
	)?;

	assert_eq!(
		tree.rectangular_layout(1.0)?.points().len(),
		num_nodes as usize
	);
	assert_eq!(
		tree.slanted_layout(1.0)?.points().len(),
		num_nodes as usize
	);
	assert_eq!(tree.tidy_layout(1.0)?.points().len(), num_nodes as usize);

	Ok(())
}

fn choose3_for_test(value: u32) -> u128 {
	let value = u128::from(value);
	value * value.saturating_sub(1) * value.saturating_sub(2) / 6
}
