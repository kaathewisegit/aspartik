use anyhow::{Context, Result, ensure};
use picoarrow::array::{ArrayUtf8, Nullable};

use super::{BinaryTree, Node};
use crate::newick::{Edge, Node as NewickNode, NodeIdx, Tree};

fn nonempty(value: &str) -> Option<&str> {
	(!value.is_empty()).then_some(value)
}

impl TryFrom<&Tree> for BinaryTree {
	type Error = anyhow::Error;

	fn try_from(tree: &Tree) -> Result<Self> {
		let num_nodes = tree.num_nodes();
		let root = *tree
			.root()
			.context("Expected a rooted Newick tree")?;
		ensure!(
			(root as usize) < num_nodes,
			"Root node {root} is out of range"
		);

		let mut children = vec![Vec::<NodeIdx>::new(); num_nodes];
		let mut parents = vec![None; num_nodes];
		let mut lengths = vec![None; num_nodes];
		let mut old_edge_metadata = vec![None; num_nodes];

		for (parent, child, edge) in tree.edge_entries() {
			ensure!(
				(parent as usize) < num_nodes,
				"Parent node {parent} is out of range"
			);
			ensure!(
				(child as usize) < num_nodes,
				"Child node {child} is out of range"
			);
			ensure!(
				parents[child as usize]
					.replace(parent)
					.is_none(),
				"Node {child} has more than one parent"
			);

			let length = edge.distance().with_context(|| {
				format!(
					"Edge from {parent} to {child} has no length"
				)
			})?;
			lengths[child as usize] = Some(length);
			old_edge_metadata[child as usize] =
				nonempty(edge.attributes());
			children[parent as usize].push(child);
		}

		ensure!(
			parents[root as usize].is_none(),
			"The root has a parent"
		);
		for node in 0..num_nodes {
			let child_count = children[node].len();
			ensure!(
				child_count == 0 || child_count == 2,
				"Node {node} has {child_count} children; expected zero or two"
			);

			if node != root as usize {
				ensure!(
					parents[node].is_some(),
					"Node {node} is not connected to a parent"
				);
			}
		}

		let mut reachable = vec![false; num_nodes];
		let mut stack = vec![root];
		while let Some(node) = stack.pop() {
			ensure!(
				!reachable[node as usize],
				"The Newick graph contains a cycle"
			);
			reachable[node as usize] = true;
			stack.extend(children[node as usize]
				.iter()
				.rev()
				.copied());
		}
		ensure!(
			reachable.iter().all(|value| *value),
			"The Newick tree contains nodes which are not reachable from the root"
		);

		let mut leaves = Vec::new();
		let mut internals = Vec::new();
		let mut stack = vec![(root, false)];
		while let Some((node, children_visited)) = stack.pop() {
			if children[node as usize].is_empty() {
				leaves.push(node);
				continue;
			}
			if children_visited {
				internals.push(node);
				continue;
			}

			stack.push((node, true));
			for &child in children[node as usize].iter().rev() {
				stack.push((child, false));
			}
		}

		ensure!(
			leaves.len() >= 2,
			"Expected at least two leaves, got {}",
			leaves.len()
		);
		let num_leaves = u32::try_from(leaves.len())?;
		let mut mapping = vec![u32::MAX; num_nodes];
		for (index, &node) in leaves.iter().enumerate() {
			mapping[node as usize] = u32::try_from(index)?;
		}
		for (index, &node) in internals.iter().enumerate() {
			mapping[node as usize] =
				num_leaves + u32::try_from(index)?;
		}
		let binary_root = mapping[root as usize];

		let mut binary_children = Vec::with_capacity(num_nodes - 1);
		for &node in &internals {
			for &child in &children[node as usize] {
				binary_children.push(mapping[child as usize]);
			}
		}

		let mut binary_lengths = vec![0.0; num_nodes - 1];
		let mut names = vec![None; num_nodes];
		let mut node_metadata = vec![None; num_nodes];
		let mut edge_metadata = vec![None; num_nodes - 1];
		for old_node in 0..num_nodes {
			let new_node = mapping[old_node] as usize;
			let node = tree.get_node(old_node as u32);
			names[new_node] = nonempty(node.name());
			node_metadata[new_node] = nonempty(node.attributes());
			if old_node != root as usize {
				let edge_index = new_node
					- usize::from(
						new_node > binary_root as usize,
					);
				binary_lengths[edge_index] = lengths[old_node]
					.context(
						"A non-root node has no edge length",
					)?;
				edge_metadata[edge_index] =
					old_edge_metadata[old_node];
			}
		}

		let mut node_names = ArrayUtf8::<Nullable>::new();
		for name in names {
			node_names.push(name)?;
		}
		let mut node_metadata_array = ArrayUtf8::<Nullable>::new();
		for metadata in node_metadata {
			node_metadata_array.push(metadata)?;
		}
		let mut edge_metadata_array = ArrayUtf8::<Nullable>::new();
		for metadata in edge_metadata {
			edge_metadata_array.push(metadata)?;
		}

		Self::new(
			num_leaves,
			binary_root,
			&binary_children,
			&binary_lengths,
			node_names,
			node_metadata_array,
			edge_metadata_array,
		)
	}
}

impl From<&BinaryTree> for Tree {
	fn from(tree: &BinaryTree) -> Self {
		let mut out = Self::new();
		let mut mapping = vec![u32::MAX; tree.num_nodes() as usize];

		for node in tree.preorder() {
			let newick_node = NewickNode::new(
				tree.name(node).unwrap_or_default().to_owned(),
				tree.node_metadata(node)
					.unwrap_or_default()
					.to_owned(),
			);
			mapping[node.index() as usize] =
				out.add_node(newick_node);
		}

		for internal in tree.internals() {
			let parent = mapping[internal.index() as usize];
			let (left, right) = tree.children_of(internal);
			for child in [left, right] {
				let length = tree.edge_length(child).expect(
					"A child node always has an incoming edge",
				);
				out.add_edge(
					parent,
					mapping[child.index() as usize],
					Edge::new(
						Some(length),
						tree.edge_metadata(child)
							.unwrap_or_default()
							.to_owned(),
					),
				);
			}
		}

		out.set_root(mapping[Node::from(tree.root()).index() as usize]);
		out
	}
}
