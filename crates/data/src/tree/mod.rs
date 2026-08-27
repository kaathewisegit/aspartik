use anyhow::{Result, anyhow, ensure};
use picoarrow::array::{Array, ArrayUtf8, Nullable};
use rustc_hash::{FxBuildHasher, FxHashSet};

use std::{
	cmp::{max, min},
	fmt,
};

mod newick;

const ROOT_PARENT: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Node(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Leaf(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Internal(u32);

impl Node {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
		self.0 as usize
	}
}

impl Leaf {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
		self.0 as usize
	}
}

impl Internal {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
		self.0 as usize
	}
}

impl From<Leaf> for Node {
	fn from(value: Leaf) -> Self {
		Self(value.0)
	}
}

impl From<Internal> for Node {
	fn from(value: Internal) -> Self {
		Self(value.0)
	}
}

pub struct BinaryRootedTree {
	num_leaves: u32,
	children: Box<[u32]>,
	parents: Box<[u32]>,
	edge_lengths: Box<[f64]>,
	node_names: ArrayUtf8<Nullable>,
	node_metadata: ArrayUtf8<Nullable>,
	edge_metadata: ArrayUtf8<Nullable>,
}

impl fmt::Debug for BinaryRootedTree {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let names: Vec<_> = self
			.nodes()
			.map(|node| self.node_names.get(node.i()))
			.collect();
		let node_metadata: Vec<_> = self
			.nodes()
			.map(|node| self.node_metadata.get(node.i()))
			.collect();
		let edge_metadata: Vec<_> = self
			.edges()
			.map(|child| self.edge_metadata.get(child.i()))
			.collect();

		f.debug_struct("BinaryRootedTree")
			.field("num_leaves", &self.num_leaves)
			.field("children", &self.children)
			.field("parents", &self.parents)
			.field("edge_lengths", &self.edge_lengths)
			.field("node_names", &names)
			.field("node_metadata", &node_metadata)
			.field("edge_metadata", &edge_metadata)
			.finish()
	}
}

impl BinaryRootedTree {
	pub fn new(
		num_leaves: u32,
		children: Box<[u32]>,
		edge_lengths: Box<[f64]>,
		mut node_names: ArrayUtf8<Nullable>,
		mut node_metadata: ArrayUtf8<Nullable>,
		mut edge_metadata: ArrayUtf8<Nullable>,
	) -> Result<Self> {
		ensure!(
			num_leaves >= 2,
			"Expected at least two leaves, got {num_leaves}"
		);

		let num_nodes = num_leaves
			.checked_mul(2)
			.and_then(|value| value.checked_sub(1))
			.ok_or_else(|| {
				anyhow!(
					"The number of nodes does not fit in u32"
				)
			})?;
		let num_edges = num_nodes - 1;
		let num_nodes_usize = usize::try_from(num_nodes)?;
		let num_edges_usize = usize::try_from(num_edges)?;

		ensure!(
			children.len() == num_edges_usize,
			"Expected {num_edges} child entries, got {}",
			children.len()
		);
		ensure!(
			edge_lengths.len() == num_edges_usize,
			"Expected {num_edges} edge lengths, got {}",
			edge_lengths.len()
		);
		ensure!(
			node_names.len() == num_nodes_usize,
			"Expected {num_nodes} node names, got {}",
			node_names.len()
		);
		ensure!(
			node_metadata.len() == num_nodes_usize,
			"Expected {num_nodes} node metadata entries, got {}",
			node_metadata.len()
		);
		ensure!(
			edge_metadata.len() == num_edges_usize,
			"Expected {num_edges} edge metadata entries, got {}",
			edge_metadata.len()
		);

		let root = num_nodes - 1;
		let mut parents = vec![ROOT_PARENT; num_nodes_usize];
		let mut seen = vec![false; num_nodes_usize];

		for (offset, pair) in
			children.as_chunks::<2>().0.iter().enumerate()
		{
			let parent = num_leaves + u32::try_from(offset)?;
			for &child in pair {
				ensure!(
					child < num_nodes,
					"Child {child} of node {parent} is out of range"
				);
				ensure!(
					child < parent,
					"Child {child} must have a lower index than parent {parent}"
				);
				ensure!(
					!seen[child as usize],
					"Node {child} appears as a child more than once"
				);

				seen[child as usize] = true;
				parents[child as usize] = parent;
			}
		}

		for node in 0..root {
			ensure!(
				seen[node as usize],
				"Node {node} is not connected to a parent"
			);
		}
		ensure!(!seen[root as usize], "The root appears as a child");

		node_names.shrink_to_fit();
		node_metadata.shrink_to_fit();
		edge_metadata.shrink_to_fit();

		Ok(Self {
			num_leaves,
			children,
			parents: parents.into_boxed_slice(),
			edge_lengths,
			node_names,
			node_metadata,
			edge_metadata,
		})
	}

	pub fn num_nodes(&self) -> u32 {
		self.num_leaves * 2 - 1
	}

	pub fn num_leaves(&self) -> u32 {
		self.num_leaves
	}

	pub fn num_internals(&self) -> u32 {
		self.num_leaves - 1
	}

	pub fn num_edges(&self) -> u32 {
		self.num_nodes() - 1
	}

	pub fn root(&self) -> Internal {
		Internal(self.num_nodes() - 1)
	}

	pub fn nodes(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		(0..self.num_nodes()).map(Node)
	}

	pub fn leaves(&self) -> impl DoubleEndedIterator<Item = Leaf> + use<> {
		(0..self.num_leaves()).map(Leaf)
	}

	pub fn internals(
		&self,
	) -> impl DoubleEndedIterator<Item = Internal> + use<> {
		(self.num_leaves()..self.num_nodes()).map(Internal)
	}

	pub fn edges(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		(0..self.num_edges()).map(Node)
	}

	pub fn is_leaf(&self, node: Node) -> bool {
		node.0 < self.num_leaves()
	}

	pub fn is_internal(&self, node: Node) -> bool {
		(self.num_leaves()..self.num_nodes()).contains(&node.0)
	}

	pub fn as_leaf(&self, node: Node) -> Option<Leaf> {
		self.is_leaf(node).then_some(Leaf(node.0))
	}

	pub fn as_internal(&self, node: Node) -> Option<Internal> {
		self.is_internal(node).then_some(Internal(node.0))
	}

	pub fn children_of(&self, node: Internal) -> (Node, Node) {
		let offset = (node.0 - self.num_leaves()) as usize * 2;
		(Node(self.children[offset]), Node(self.children[offset + 1]))
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		let parent = self.parents[node.i()];
		(parent != ROOT_PARENT).then_some(Internal(parent))
	}

	pub fn edge_length(&self, child: Node) -> Option<f64> {
		(child.0 < self.num_edges())
			.then(|| self.edge_lengths[child.i()])
	}

	pub fn name(&self, node: Node) -> Option<&str> {
		self.node_names.get(node.i())
	}

	pub fn node_metadata(&self, node: Node) -> Option<&str> {
		self.node_metadata.get(node.i())
	}

	pub fn edge_metadata(&self, child: Node) -> Option<&str> {
		(child.0 < self.num_edges())
			.then(|| self.edge_metadata.get(child.i()))
			.flatten()
	}

	pub fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		self.leaves().find(|leaf| {
			self.node_names.get(leaf.0 as usize) == Some(name)
		})
	}

	pub fn ola(&self) -> Vec<i32> {
		let num_nodes = self.num_nodes();
		let num_leaves = self.num_leaves();

		let mut labels = Vec::from_iter(0..num_leaves as i32);
		labels.resize(num_nodes as usize, 0);

		let mut clade_founder = vec![0; num_nodes as usize];

		for node in self.postorder() {
			if let Some(leaf) = self.as_leaf(node) {
				clade_founder[leaf.i()] = leaf.0
			} else if let Some(internal) = self.as_internal(node) {
				let (left, right) = self.children_of(internal);
				clade_founder[internal.i()] = min(
					clade_founder[left.i()],
					clade_founder[right.i()],
				);
			} else {
				unreachable!()
			}
		}

		let mut clade_splitter = vec![0; num_nodes as usize];
		let mut splitter_to_node = vec![0; num_leaves as usize];
		for node in self.preorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};

			let (left, right) = self.children_of(internal);

			let splitter = max(
				clade_founder[left.i()],
				clade_founder[right.i()],
			);
			clade_splitter[node.i()] = splitter;
			labels[node.i()] = -(splitter as i32);
			splitter_to_node[splitter as usize] = node.0;
		}

		let mut ola = Vec::new();
		let mut forward_to = Vec::from_iter(0..num_nodes);

		for label in (1..num_leaves).rev() {
			let splitter_node =
				Internal(splitter_to_node[label as usize]);
			let (left, right) = self.children_of(splitter_node);

			let sibling = if clade_founder[left.i()] == label {
				right
			} else {
				left
			};

			let mut curr = sibling;
			while forward_to[curr.i()] != curr.0 {
				curr = Node(forward_to[curr.i()]);
			}

			ola.push(labels[curr.i()]);
			forward_to[splitter_node.i()] = curr.0;
		}
		ola.reverse();

		ola
	}

	pub fn robinson_foulds(&self, other: &Self) -> u32 {
		let mut counter = 0;
		let mut labels = vec![0; self.num_nodes() as usize];
		let mut clades = FxHashSet::with_capacity_and_hasher(
			self.num_nodes() as usize,
			FxBuildHasher,
		);

		fn process(
			node: Node,
			tree: &BinaryRootedTree,
			counter: &mut u32,
			labels: &mut [u32],
			clades: &mut FxHashSet<(u32, u32)>,
		) -> (u32, u32, u32) {
			let Some(internal) = tree.as_internal(node) else {
				let label = *counter;
				labels[node.i()] = label;
				*counter += 1;
				return (label, label, 1);
			};

			let (left, right) = tree.children_of(internal);
			let (left_min, _left_max, left_size) =
				process(left, tree, counter, labels, clades);
			let (_right_min, right_max, right_size) =
				process(right, tree, counter, labels, clades);
			let size = left_size + right_size;

			if node != tree.root().into() {
				clades.insert((left_min, right_max));
			}

			(left_min, right_max, size)
		}
		process(
			self.root().into(),
			self,
			&mut counter,
			&mut labels,
			&mut clades,
		);

		let mut num_shared_clades = 0;
		fn process_other(
			node: Node,
			tree: &BinaryRootedTree,
			labels: &[u32],
			clades: &FxHashSet<(u32, u32)>,
			num_shared: &mut u32,
		) -> (u32, u32, u32) {
			let Some(internal) = tree.as_internal(node) else {
				let label = labels[node.i()];
				return (label, label, 1);
			};

			let (left, right) = tree.children_of(internal);
			let (left_min, left_max, left_size) = process_other(
				left, tree, labels, clades, num_shared,
			);
			let (right_min, right_max, right_size) = process_other(
				right, tree, labels, clades, num_shared,
			);
			let size = left_size + right_size;

			let min_val = left_min.min(right_min);
			let max_val = left_max.max(right_max);

			if node != tree.root().into()
				&& max_val - min_val + 1 == size
				&& clades.contains(&(min_val, max_val))
			{
				*num_shared += 1;
			}

			(min_val, max_val, size)
		}
		process_other(
			other.root().into(),
			other,
			&labels,
			&clades,
			&mut num_shared_clades,
		);

		let num_nontrivial = self.num_leaves() - 2;
		2 * (num_nontrivial - num_shared_clades)
	}

	pub fn preorder(&self) -> impl Iterator<Item = Node> + '_ {
		Preorder {
			tree: self,
			stack: vec![self.root().into()],
		}
	}

	pub fn postorder(&self) -> impl Iterator<Item = Node> + '_ {
		Postorder {
			tree: self,
			stack: vec![(self.root().into(), false)],
		}
	}
}

struct Preorder<'a> {
	tree: &'a BinaryRootedTree,
	stack: Vec<Node>,
}

impl Iterator for Preorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Self::Item> {
		let node = self.stack.pop()?;

		if let Some(internal) = self.tree.as_internal(node) {
			let (left, right) = self.tree.children_of(internal);
			self.stack.push(right);
			self.stack.push(left);
		}

		Some(node)
	}
}

struct Postorder<'a> {
	tree: &'a BinaryRootedTree,
	stack: Vec<(Node, bool)>,
}

impl Iterator for Postorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Self::Item> {
		while let Some((node, children_visited)) = self.stack.pop() {
			if children_visited {
				return Some(node);
			}

			self.stack.push((node, true));
			if let Some(internal) = self.tree.as_internal(node) {
				let (left, right) =
					self.tree.children_of(internal);
				self.stack.push((right, false));
				self.stack.push((left, false));
			}
		}

		None
	}
}
