use anyhow::{Result, ensure};
use picoarrow::array::{Array, ArrayUtf8, Nullable};
use rand::{Rng, seq::SliceRandom};

use std::{
	cmp::{Reverse, max, min},
	collections::BinaryHeap,
	slice,
};

use super::{Internal, Leaf, Node, TOMBSTONE, Tree};
use crate::TaxonSet;
use computare_buffer::Buffer;

#[derive(Debug)]
pub struct BinaryTree {
	num_leaves: u32,
	root: u32,
	children: Buffer<u32>,
	parents: Buffer<u32>,
	edge_lengths: Buffer<f64>,
	taxa: TaxonSet,
	internal_names: ArrayUtf8<Nullable>,
}

fn validate_num_leaves(num_leaves: u32) -> Result<()> {
	ensure!(
		num_leaves >= 2,
		"A tree cannot have less than two leaves, got {num_leaves}"
	);
	ensure!(
		num_leaves <= u32::MAX / 2,
		"A tree cannot have more than {} leaves, got {num_leaves}",
		u32::MAX / 2
	);
	Ok(())
}

impl BinaryTree {
	pub fn random<R: Rng>(taxa: TaxonSet, rng: &mut R) -> Result<Self> {
		let num_leaves = u32::try_from(taxa.len())?;
		validate_num_leaves(num_leaves)?;
		let num_nodes = num_leaves * 2 - 1;
		let num_internals = num_leaves - 1;
		let num_nodes_usize = usize::try_from(num_nodes)?;
		let num_edges_usize = num_nodes_usize - 1;

		let internals = num_leaves..num_nodes;
		let mut parents = Buffer::repeat(TOMBSTONE, num_nodes_usize);
		for (slot, parent) in parents[..num_edges_usize]
			.iter_mut()
			.zip(internals.clone().chain(internals))
		{
			*slot = parent;
		}
		let root = parents[num_edges_usize - 1];
		parents[..num_edges_usize - 1].shuffle(rng);

		let mut children = Buffer::repeat(TOMBSTONE, num_edges_usize);
		let mut remaining = vec![2; num_internals as usize];
		*remaining.last_mut().unwrap() = 1;
		let mut unused =
			BinaryHeap::from_iter((0..num_leaves).map(Reverse));
		let mut removed = Vec::with_capacity(num_edges_usize);

		for step in 0..num_edges_usize - 1 {
			let parent = parents[step];
			let child = unused.pop().unwrap().0;
			removed.push(child);
			let offset = ((parent - num_leaves) * 2) as usize;
			if children[offset] == TOMBSTONE {
				children[offset] = child;
			} else {
				children[offset + 1] = child;
			}

			let index = (parent - num_leaves) as usize;
			remaining[index] -= 1;
			if remaining[index] == 0 {
				unused.push(Reverse(parent));
			}
		}

		let child = unused.pop().unwrap().0;
		removed.push(child);
		children[(root - num_leaves) as usize * 2 + 1] = child;
		for index in 0..num_edges_usize {
			while removed[index] != index as u32 {
				let target = removed[index] as usize;
				parents.swap(index, target);
				removed.swap(index, target);
			}
		}

		let mut internal_names = ArrayUtf8::<Nullable>::new();
		internal_names.push_nulls(num_internals as usize);
		let edge_lengths = Buffer::repeat(1.0, num_nodes_usize);

		Ok(Self {
			num_leaves,
			root,
			children,
			parents,
			edge_lengths,
			taxa,
			internal_names,
		})
	}

	pub fn new(
		num_leaves: u32,
		root: u32,
		children: Buffer<u32>,
		parents: Buffer<u32>,
		edge_lengths: Buffer<f64>,
		taxa: TaxonSet,
		internal_names: ArrayUtf8<Nullable>,
	) -> Self {
		let mut tree = Self {
			num_leaves,
			root,
			children,
			parents,
			edge_lengths,
			taxa,
			internal_names,
		};
		tree.internal_names.shrink_to_fit();
		tree
	}

	pub fn validate(&self) -> Result<()> {
		let num_leaves = self.num_leaves;
		let root = self.root;

		validate_num_leaves(num_leaves)?;
		let num_nodes = num_leaves * 2 - 1;
		let num_edges = num_nodes - 1;
		let num_nodes_usize = num_nodes as usize;
		let num_edges_usize = num_edges as usize;
		ensure!(
			(num_leaves..num_nodes).contains(&root),
			"Root node {root} is not an internal node"
		);
		ensure!(
			self.children.len() == num_edges_usize,
			"Expected {num_edges} child entries, got {}",
			self.children.len()
		);
		ensure!(
			self.parents.len() == num_nodes_usize,
			"Expected {num_nodes} parent entries, got {}",
			self.parents.len()
		);
		ensure!(
			self.edge_lengths.len() == num_nodes_usize,
			"Expected {num_nodes} edge lengths, got {}",
			self.edge_lengths.len()
		);
		ensure!(
			self.taxa.len() == num_leaves as usize,
			"Expected {num_leaves} taxa, got {}",
			self.taxa.len()
		);
		ensure!(
			self.internal_names.len() == (num_leaves - 1) as usize,
			"Expected {} internal names, got {}",
			num_leaves - 1,
			self.internal_names.len()
		);

		let mut seen = vec![false; num_nodes_usize];
		for (offset, pair) in
			self.children.as_chunks::<2>().0.iter().enumerate()
		{
			let parent = num_leaves + u32::try_from(offset)?;
			for &child in pair {
				ensure!(
					child < num_nodes,
					"Child {child} of node {parent} is out of range"
				);
				ensure!(
					!seen[child as usize],
					"Node {child} appears as a child more than once"
				);
				ensure!(
					self.parents[child as usize] == parent,
					"Parent of node {child} is inconsistent"
				);
				seen[child as usize] = true;
			}
		}

		for node in 0..num_nodes {
			if node == root {
				ensure!(
					!seen[node as usize],
					"The root appears as a child"
				);
				ensure!(
					self.parents[node as usize]
						== TOMBSTONE,
					"The root has a parent"
				);
			} else {
				ensure!(
					seen[node as usize],
					"Node {node} is not connected to a parent"
				);
			}
		}

		seen.fill(false);
		let mut stack = vec![root];
		while let Some(node) = stack.pop() {
			ensure!(
				!seen[node as usize],
				"The tree contains a cycle"
			);
			seen[node as usize] = true;
			if node >= num_leaves {
				let offset = (node - num_leaves) as usize * 2;
				stack.extend_from_slice(
					&self.children[offset..offset + 2],
				);
			}
		}
		ensure!(
			seen.iter().all(|value| *value),
			"Not all nodes are reachable from the root"
		);

		Ok(())
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
		Internal(self.root)
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
		let root = self.root;
		(0..root).chain(root + 1..self.num_nodes()).map(Node)
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

	pub fn children_of(&self, node: Internal) -> [Node; 2] {
		let offset = (node.0 - self.num_leaves()) as usize * 2;
		[Node(self.children[offset]), Node(self.children[offset + 1])]
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		let parent = self.parents[node.usize()];
		(parent != TOMBSTONE).then_some(Internal(parent))
	}

	pub fn edge_length(&self, child: Node) -> Option<f64> {
		if child.0 == self.root {
			None
		} else {
			Some(self.edge_lengths[child.usize()])
		}
	}

	pub fn taxa(&self) -> &TaxonSet {
		&self.taxa
	}

	pub fn set_taxa(&mut self, taxa: TaxonSet) {
		self.taxa = taxa;
	}

	pub fn name(&self, node: Node) -> Option<&str> {
		if node.0 < self.num_leaves {
			let name = self.taxa.get(node.usize());
			(!name.is_empty()).then_some(name)
		} else {
			let index = node.usize() - self.num_leaves as usize;
			if index < self.internal_names.len() {
				self.internal_names.get(index)
			} else {
				None
			}
		}
	}

	pub fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		if name.is_empty() {
			return None;
		}
		self.leaves()
			.find(|leaf| self.taxa.get(leaf.usize()) == name)
	}

	pub fn ola(&self) -> Vec<i32> {
		let num_nodes = self.num_nodes();
		let num_leaves = self.num_leaves();

		let mut labels = Vec::from_iter(0..num_leaves as i32);
		labels.resize(num_nodes as usize, 0);

		let mut clade_founder = vec![0; num_nodes as usize];

		for node in self.postorder() {
			if let Some(leaf) = self.as_leaf(node) {
				clade_founder[leaf.usize()] = leaf.0
			} else if let Some(internal) = self.as_internal(node) {
				let [left, right] = self.children_of(internal);
				clade_founder[internal.usize()] = min(
					clade_founder[left.usize()],
					clade_founder[right.usize()],
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

			let [left, right] = self.children_of(internal);

			let splitter = max(
				clade_founder[left.usize()],
				clade_founder[right.usize()],
			);
			clade_splitter[node.usize()] = splitter;
			labels[node.usize()] = -(splitter as i32);
			splitter_to_node[splitter as usize] = node.0;
		}

		let mut ola = Vec::new();
		let mut forward_to = Vec::from_iter(0..num_nodes);

		for label in (1..num_leaves).rev() {
			let splitter_node =
				Internal(splitter_to_node[label as usize]);
			let [left, right] = self.children_of(splitter_node);

			let sibling = if clade_founder[left.usize()] == label {
				right
			} else {
				left
			};

			let mut curr = sibling;
			while forward_to[curr.usize()] != curr.0 {
				curr = Node(forward_to[curr.usize()]);
			}

			ola.push(labels[curr.usize()]);
			forward_to[splitter_node.usize()] = curr.0;
		}
		ola.reverse();

		ola
	}

	/// Returns `true` if the children have the same names and ids
	pub fn identical_children(&self, other: &BinaryTree) -> bool {
		self.taxa == other.taxa
	}
}

impl Tree for BinaryTree {
	fn root(&self) -> Option<Node> {
		Some(Self::root(self).into())
	}

	fn parent_of(&self, node: Node) -> Option<Node> {
		self.parent_of(node).as_deref().copied()
	}

	fn children_of(&self, node: Node) -> &[Node] {
		if self.is_leaf(node) {
			return &[];
		}
		let offset = (node.0 - self.num_leaves()) as usize * 2;
		let slice = &self.children[offset..offset + 2];
		// SAFETY: Node is a transparent wrapper over u32
		let slice: &[Node] = unsafe {
			slice::from_raw_parts(
				slice.as_ptr() as *const Node,
				slice.len(),
			)
		};
		slice
	}

	fn name_of(&self, node: Node) -> &str {
		self.name(node).unwrap_or("")
	}

	fn edge_length(&self, node: Node) -> Option<f64> {
		self.edge_length(node)
	}
}
