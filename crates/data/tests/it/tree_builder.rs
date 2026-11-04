use arbitrary::{Arbitrary, Unstructured};
use arbtest::arbtest;

use std::collections::HashSet;

use data::tree::{Node, TOMBSTONE, TreeBuilder};

#[test]
fn build_basic() {
	let mut builder = TreeBuilder::new();

	let root = builder.root();

	let left = builder.add_node("left", 1.0, root);
	let right = builder.add_node("right", 2.5, root);
	let left_left = builder.add_node("a", 0.5, left);
	let left_right = builder.add_node("b", 1.5, left);

	assert_eq!(builder.num_nodes(), 5);
	assert_eq!(builder.root(), root);

	assert_eq!(builder.parent_of(root), None);
	assert_eq!(builder.parent_of(left), Some(root));
	assert_eq!(builder.parent_of(right), Some(root));
	assert_eq!(builder.parent_of(left_left), Some(left));
	assert_eq!(builder.parent_of(left_right), Some(left));

	assert_eq!(builder.children_of(root), &[left, right]);
	assert_eq!(builder.children_of(left), &[left_left, left_right]);
	assert!(builder.children_of(right).is_empty());
	assert!(builder.is_leaf(right));
	assert!(!builder.is_leaf(left));

	assert_eq!(builder.name(left_left), "a");
	assert_eq!(builder.edge_length(left_left), Some(0.5));
	assert_eq!(builder.edge_length(root), None);
	assert_eq!(builder.edge_length(right), Some(2.5));

	builder.set_node_name(right, "c");
	assert_eq!(builder.name(right), "c");
	builder.set_edge_length(right, 3.0);
	assert_eq!(builder.edge_length(right), Some(3.0));

	assert_eq!(builder.nodes().count(), 5);
	assert!(builder.nodes().any(|n| n == left_right));

	builder.clear();
	assert_eq!(builder.num_nodes(), 0);
	assert_eq!(builder.root(), Node(TOMBSTONE));
}

#[test]
fn build_arbitrary() {
	arbtest(|u: &mut Unstructured<'_>| {
		let mut builder = TreeBuilder::new();

		let root = builder.root();

		let num_steps = u.int_in_range(0..=1_000u32)?;
		for _ in 0..num_steps {
			let num_nodes = builder.num_nodes();
			let parent = Node(u.int_in_range(0..=(num_nodes - 1))?);

			let name = u64::arbitrary(u)?.to_string();
			let edge_length = f64::from(u16::arbitrary(u)?);
			let child =
				builder.add_node(&name, edge_length, parent);

			assert_eq!(builder.parent_of(child), Some(parent));
			assert_eq!(builder.name(child), name);
			assert_eq!(
				builder.edge_length(child),
				Some(edge_length)
			);
			assert!(builder.children_of(child).is_empty());
			assert!(builder.children_of(parent).contains(&child));
		}

		assert_eq!(builder.root(), root);
		assert_eq!(builder.num_nodes(), num_steps + 1);
		assert_eq!(
			builder.nodes().count(),
			builder.num_nodes() as usize,
		);
		assert_eq!(builder.parent_of(root), None);

		let mut traversed = HashSet::new();
		for node in builder.nodes() {
			let mut current = node;
			while let Some(parent) = builder.parent_of(current) {
				traversed.insert((parent, current));
				current = parent;
			}
			assert_eq!(current, root);
		}
		assert_eq!(traversed.len(), num_steps as usize);

		Ok(())
	})
	.size_min(2u32.pow(20));
}
