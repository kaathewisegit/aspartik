use anyhow::{Context, Result, bail, ensure};
use bytemuck::allocation::cast_vec;
use computare_buffer::Buffer;
use parking_lot::Mutex;
use picoarrow::array::{Array, ArrayUtf8, Nullable};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};
use rand::{RngExt, seq::SliceRandom};

use std::{
	cmp::{max, min},
	collections::VecDeque,
	io::Write,
	slice,
};

use crate::{clock::PyClock, impl_pyparameter_common, parameters::Parameter};
use bitmap::Bitmap;
use data::{
	PyTaxonSet, TaxonSet,
	tree::{
		BinaryTree, Internal, Leaf, Node, TOMBSTONE, Tree, TreeBuilder,
		python::PyBinaryTree, python::PyTreeBuilder,
	},
};
use rng::{PyRng, Rng};
use sk::EpochBuf;
use util::{atomic::MonotonicU32, py_bail};

#[derive(Debug, Clone)]
pub struct DatedTree {
	names: TaxonSet,

	num_nodes: u32,
	num_leaves: u32,
	num_internals: u32,

	/// `(value, backup)`
	root: (u32, u32),

	children: EpochBuf<u32>,
	parents: EpochBuf<u32>,
	heights: EpochBuf<f64>,

	updated_edges: Bitmap,
	/// A bitmap of internal nodes whose child edges got updated
	updated_partials: Bitmap,
	/// A bitmap of nodes whose parents were updated
	updated_propagatons: Bitmap,
}

impl DatedTree {
	pub fn new(names: TaxonSet, rng: &mut Rng) -> Result<Self> {
		ensure!(
			names.len() >= 2,
			"Expected at least 2 nodes, got {}",
			names.len()
		);

		let num_leaves = names.len();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;

		let mut out = Self {
			names,

			num_nodes: num_nodes as u32,
			num_leaves: num_leaves as u32,
			num_internals: num_internals as u32,

			root: (0, 0),

			children: EpochBuf::repeat(
				TOMBSTONE,
				num_internals * 2,
			),
			parents: EpochBuf::repeat(TOMBSTONE, num_nodes),
			heights: EpochBuf::repeat(0.0, num_nodes),

			updated_edges: Bitmap::new(num_nodes),
			updated_partials: Bitmap::new(num_nodes),
			updated_propagatons: Bitmap::new(num_nodes),
		};

		out.set_random_edges(rng);
		out.set_random_heights(0.01, rng);
		Ok(out)
	}

	pub fn set_random_edges(&mut self, rng: &mut Rng) {
		let num_leaves = self.num_leaves();
		let num_internals = self.num_internals();
		let num_nodes = self.num_nodes();
		// Here we create a Prüfer sequence, which encodes a binary tree
		// with the root in the last node with the ID `2l - 2`.  To do
		// that we create a sequence in which all internal nodes appear
		// twice.  Except the last node, which only appears once.
		let internals = num_leaves..num_nodes;
		let mut prüfer: Vec<u32> =
			internals.clone().chain(internals).collect();
		let root = prüfer.pop().unwrap(); // remove the last node
		prüfer.shuffle(rng); // random shuffle

		let mut parents = vec![TOMBSTONE; num_nodes as usize];
		let mut children =
			vec![TOMBSTONE; (2 * num_internals) as usize];

		let mut degrees = vec![1u8; num_nodes as usize];
		for internal in num_leaves..num_nodes {
			degrees[internal as usize] = 3;
		}
		degrees[root as usize] = 2;

		let mut ptr = 0;
		let mut next = ptr;
		for parent in prüfer {
			let child = next;

			parents[child as usize] = parent;

			// first encountered child goes in the left slot, second
			// one goes in the right
			let idx = ((parent - num_leaves) * 2) as usize
				+ degrees[parent as usize] as usize
				- 2;
			children[idx] = child;

			degrees[parent as usize] -= 1;
			degrees[child as usize] -= 1;

			if degrees[parent as usize] == 1 && parent < ptr {
				next = parent;
			} else {
				while degrees[ptr as usize] != 1
					&& ptr < num_nodes
				{
					ptr += 1;
				}
				next = ptr;
			}
		}
		let child = next;
		parents[child as usize] = root;
		children[(root - num_leaves) as usize * 2 + 1] = child;

		self.parents = parents.into();
		self.children = children.into();
		self.root = (root, root);
	}

	/// Sets the heights of internal nodes by walking upwards breadth-first
	/// starting with all of the leaves
	pub fn set_random_heights(&mut self, diff: f64, rng: &mut Rng) {
		let mut heights = vec![0.0; self.num_nodes() as usize];
		for leaf in self.leaves() {
			heights[leaf.usize()] = self.height_of(*leaf);
		}
		for node in self.postorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};

			let (left, right) = self.children_of(internal);
			let max = f64::max(
				heights[left.usize()],
				heights[right.usize()],
			);

			let node_diff = diff * (1.0 + rng.random::<f64>());
			heights[internal.usize()] = max + node_diff;
		}
		for node in self.nodes() {
			self.set_height(node, heights[node.usize()]);
		}

		self.accept();
	}

	pub fn load_newick(&mut self, newick: &TreeBuilder) -> Result<()> {
		let tree = newick.to_binary()?;
		ensure!(tree.num_leaves() == self.num_leaves());
		ensure!(tree.num_nodes() == self.num_nodes());

		let mut mapping = vec![TOMBSTONE; tree.num_nodes() as usize];
		let mut seen = vec![false; self.num_leaves() as usize];
		for leaf in tree.leaves() {
			let node = Node::from(leaf);
			let name = tree.name(node).unwrap_or_default();
			let index = self
				.names()
				.iter()
				.position(|candidate| candidate == name)
				.with_context(|| {
					format!("Node {name} not found in tree")
				})?;
			ensure!(
				!seen[index],
				"Node {name} occurs more than once"
			);
			seen[index] = true;
			mapping[node.usize()] = index as u32;
		}
		ensure!(seen.into_iter().all(|value| value));

		let mut internal_index = self.num_leaves();
		for node in tree.postorder() {
			if tree.is_internal(node) {
				mapping[node.usize()] = internal_index;
				internal_index += 1;
			}
		}

		for index in 0..self.num_edges() as usize {
			self.children[index] = TOMBSTONE;
		}
		for index in 0..self.num_nodes() as usize {
			self.parents[index] = TOMBSTONE;
		}
		for internal in tree.internals() {
			let parent = mapping[internal.usize()];
			let offset = (parent - self.num_leaves()) as usize * 2;
			let [left, right] = tree.children_of(internal);
			for (slot, child) in
				[left, right].into_iter().enumerate()
			{
				let child = mapping[child.usize()];
				self.children[offset + slot] = child;
				self.parents[child as usize] = parent;
			}
		}
		let root = mapping[tree.root().usize()];
		self.set_root(Internal(root));

		let mut heights = vec![0.0; tree.num_nodes() as usize];
		for node in tree.preorder() {
			let Some(parent) = tree.parent_of(node) else {
				continue;
			};
			let length = tree.edge_length(node).context(
				"Encountered Newick node without length",
			)?;
			heights[node.usize()] =
				heights[parent.usize()] - length;
		}
		let minimum = heights.iter().copied().fold(0.0, f64::min);
		for node in tree.nodes() {
			self.set_height(
				Node(mapping[node.usize()]),
				heights[node.usize()] - minimum,
			);
		}

		Ok(())
	}

	/// Ordered leaf attachement vector
	///
	/// From [Vector Encoding of Phylogenetic Trees by Ordered Leaf
	/// Attachment][doi].
	///
	/// [doi]: https://doi.org/10.1007/s11538-026-01611-9
	pub fn ola(&self) -> Vec<i32> {
		let num_nodes = self.num_nodes();
		let num_leaves = self.num_leaves(); // n in the paper

		let mut labels = Vec::from_iter(0..num_leaves as i32);
		labels.resize(num_nodes as usize, 0);

		let mut clade_founder = vec![0; num_nodes as usize];

		for node in self.postorder() {
			if let Some(leaf) = self.as_leaf(node) {
				clade_founder[leaf.usize()] = leaf.0
			} else if let Some(internal) = self.as_internal(node) {
				let (left, right) = self.children_of(internal);
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

			let (left, right) = self.children_of(internal);

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
			let (left, right) = self.children_of(splitter_node);

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

	pub fn names(&self) -> &TaxonSet {
		&self.names
	}

	pub fn scale(&mut self, scale: f64) -> Result<u32> {
		for node in self.internals() {
			let new_height = self.height_of(*node) * scale;
			self.set_height(*node, new_height);
		}

		if self.has_dated_tips() && scale < 1.0 {
			for node in self.nodes() {
				ensure!(self.is_node_height_valid(node));
			}
		}

		Ok(self.num_internals())
	}

	fn clear_updated(&mut self) {
		self.updated_edges.set_all_off();
		self.updated_partials.set_all_off();
		self.updated_propagatons.set_all_off();
	}

	/// Update internal edge and node bookkeeping
	pub fn mark_edge_updated(&mut self, edge: u32) {
		self.updated_edges.set_on(edge as usize);
		let (child, _) = self.edge_nodes(edge);

		self.updated_propagatons.set_on(child.usize());

		let mut curr = child;
		while let Some(next) = self.parent_of(curr) {
			curr = *next;

			if self.updated_partials.at(curr.usize()) {
				break;
			}

			self.updated_propagatons.set_on(curr.usize());
			self.updated_partials.set_on(curr.usize());
		}
	}

	/// Mark all edges as requiring updates
	///
	/// Also marks all nodes as updated.  The latter is needed because the
	/// clock and the substituion model might call this method via
	/// `Transitions`.  And it might make sense to the requirement to mark
	/// nodes in the future, instead computing them from edges.
	pub fn mark_all_edges_updated(&mut self) {
		self.updated_edges.set_all_on();
		self.updated_partials.set_all_on();
		self.updated_propagatons.set_all_on();
	}

	pub fn edges_to_update(&self) -> Vec<u32> {
		let mut out = Vec::new();
		for edge in self.edges() {
			if self.is_edge_updated(edge) {
				out.push(edge);
			}
		}
		out
	}

	fn updated_internals(&self, bitmap: &Bitmap) -> Vec<Internal> {
		let mut internals = Vec::from([self.root()]);
		let mut queue = VecDeque::from([self.root()]);

		while let Some(node) = queue.pop_front() {
			let (left, right) = self.children_of(node);

			if bitmap.at(left.usize())
				&& let Some(left) = self.as_internal(left)
			{
				internals.push(left);
				queue.push_back(left);
			}
			if bitmap.at(right.usize())
				&& let Some(right) = self.as_internal(right)
			{
				internals.push(right);
				queue.push_back(right);
			}
		}

		internals.reverse();
		internals
	}

	fn children_list(&self, internals: &[Internal]) -> Vec<[Node; 2]> {
		let mut children = Vec::with_capacity(internals.len() * 2);
		for &internal in internals {
			let (left, right) = self.children_of(internal);
			if left < right {
				children.push([left, right]);
			} else {
				children.push([right, left]);
			}
		}
		children
	}

	pub fn propagation_lists(&self) -> (Vec<u32>, Vec<[u32; 2]>, usize) {
		let mut nodes = Vec::<u32>::new();

		// Updated leaves, in order
		for leaf in self.leaves() {
			if self.updated_propagatons.at(leaf.usize()) {
				nodes.push(leaf.u32());
			}
		}
		let num_updated_leaves = nodes.len();

		let internals =
			self.updated_internals(&self.updated_propagatons);
		let children = self.children_list(&internals);

		nodes.append(&mut cast_vec(internals));

		(cast_vec(nodes), cast_vec(children), num_updated_leaves)
	}

	pub fn partials_lists(&self) -> (Vec<u32>, Vec<[u32; 2]>) {
		let internals = self.updated_internals(&self.updated_partials);
		let children = self.children_list(&internals);

		(cast_vec(internals), cast_vec(children))
	}

	fn is_edge_updated(&self, edge: u32) -> bool {
		self.updated_edges.at(edge as usize)
	}

	pub fn changed_height_nodes(&self) -> &[usize] {
		self.heights.changed_indices()
	}

	pub fn replace_child(
		&mut self,
		parent: Internal,
		old_child: Node,
		new_child: Node,
	) -> Result<()> {
		self.parents[new_child.usize()] = parent.0;

		let (left, right) = self.children_of(parent);
		let idx = (parent.0 - self.num_leaves()) as usize * 2;

		if old_child == left {
			self.children[idx] = new_child.0;
		} else if old_child == right {
			self.children[idx + 1] = new_child.0;
		} else {
			bail!(
				"replace_child: {old_child:?} is not a child of {parent:?}"
			);
		}

		self.mark_edge_updated(self.edge_index(new_child));

		Ok(())
	}

	/// Prunes parent of `node` and grafts it as a parent of `other`
	pub fn spr(&mut self, node: Node, other: Node) -> Result<()> {
		let parent = self.parent_of(node).unwrap();
		let grandparent = self.parent_of(*parent).unwrap();
		let other_parent = self.parent_of(other).unwrap();

		let sibling = self.other_child(parent, node)?;

		self.replace_child(grandparent, *parent, sibling)?;

		self.replace_child(other_parent, other, *parent)?;

		self.replace_child(parent, sibling, other)?;

		Ok(())
	}

	/// Sets the height of `node`, recording it and it's parent and child
	/// edges (if it has those).
	pub fn set_height(&mut self, node: Node, height: f64) {
		self.heights[node.usize()] = height;

		if self.parent_of(node).is_some() {
			self.mark_edge_updated(self.edge_index(node));
		}
		if let Some(node) = self.as_internal(node) {
			let (left, right) = self.children_of(node);
			self.mark_edge_updated(self.edge_index(left));
			self.mark_edge_updated(self.edge_index(right));
		}
	}

	/// Doesn't overwrite the old root.
	pub fn set_root(&mut self, node: Internal) {
		self.root.0 = node.0;
		self.parents[node.usize()] = TOMBSTONE;
	}

	pub fn swap_parents(&mut self, a: Node, b: Node) -> Result<()> {
		let Some(a_parent) = self.parent_of(a) else {
			bail!("a must not be root");
		};
		let Some(b_parent) = self.parent_of(b) else {
			bail!("b must not be root");
		};

		self.replace_child(a_parent, a, b)?;
		self.replace_child(b_parent, b, a)?;

		ensure!(self.is_node_height_valid(a));
		ensure!(self.is_node_height_valid(b));

		Ok(())
	}

	pub(crate) fn is_node_height_valid(&self, node: Node) -> bool {
		let height = self.height_of(node);

		if let Some(parent) = self.parent_of(node)
			&& height >= self.height_of(*parent)
		{
			return false;
		}

		let Some(internal) = self.as_internal(node) else {
			return true;
		};

		let (left, right) = self.children_of(internal);

		height > self.height_of(left) && height > self.height_of(right)
	}

	fn has_dated_tips(&self) -> bool {
		for leaf in self.leaves() {
			if self.height_of(*leaf) != 0.0 {
				return true;
			}
		}

		false
	}

	pub fn total_length(&self) -> f64 {
		self.edges().map(|e| self.edge_length(e)).sum()
	}

	pub fn validate(&self) -> Result<()> {
		for (i, parent) in self.parents.iter().enumerate() {
			ensure!(
				*parent >= self.num_leaves(),
				"Leaf {} became a parent of {}",
				parent,
				i
			)
		}

		for node in self.internals() {
			let (left, right) = self.children_of(node);

			ensure!(
				self.height_of(*node) > self.height_of(left),
				"Node {} ({}) is younger than it's left child {} ({})",
				node.0,
				self.height_of(*node),
				left.0,
				self.height_of(left),
			);
			ensure!(
				self.height_of(*node) > self.height_of(right),
				"Node {} ({}) is younger than it's right child {} ({})",
				node.0,
				self.height_of(*node),
				right.0,
				self.height_of(right),
			);

			let left_parent = self.parent_of(left);
			let right_parent = self.parent_of(right);
			ensure!(
				left_parent.is_some_and(|p| p == node),
				"Expected {left:?} to have the parent {node:?}, got {left_parent:?}"
			);
			ensure!(
				right_parent.is_some_and(|p| p == node),
				"Expected {right:?} to have the parent {node:?}, got {right_parent:?}"
			);
		}

		let roots: Vec<u32> = self
			.parents
			.iter()
			.enumerate()
			.filter(|&(_, &parent)| parent == TOMBSTONE)
			.map(|(idx, _)| idx as u32)
			.collect();
		ensure!(
			roots.len() == 1,
			"The tree has more than one root: {:?}",
			roots
		);
		ensure!(roots[0] == self.root.0);

		let mut is_child = vec![false; self.num_nodes() as usize];
		for node in self.internals() {
			let (left, right) = self.children_of(node);
			is_child[left.usize()] = true;
			is_child[right.usize()] = true;
		}
		let child_count: u32 =
			is_child.into_iter().map(u32::from).sum();
		ensure!(child_count == self.num_nodes() - 1);

		Ok(())
	}

	pub fn num_nodes(&self) -> u32 {
		self.num_nodes
	}

	pub fn num_internals(&self) -> u32 {
		self.num_internals
	}

	pub fn num_leaves(&self) -> u32 {
		self.num_leaves
	}

	pub fn num_edges(&self) -> u32 {
		self.num_internals() * 2
	}

	pub fn is_internal(&self, node: Node) -> bool {
		node.0 >= self.num_leaves()
	}

	pub fn is_leaf(&self, node: Node) -> bool {
		node.0 < self.num_leaves()
	}

	pub fn as_internal(&self, node: Node) -> Option<Internal> {
		if self.is_internal(node) {
			Some(Internal(node.0))
		} else {
			None
		}
	}

	pub fn as_leaf(&self, node: Node) -> Option<Leaf> {
		if self.is_leaf(node) {
			Some(Leaf(node.0))
		} else {
			None
		}
	}

	pub fn root(&self) -> Internal {
		Internal(self.root.0)
	}

	pub fn height_of(&self, node: Node) -> f64 {
		self.heights[node.usize()]
	}

	pub fn children_of(&self, node: Internal) -> (Node, Node) {
		let offset = (node.0 - self.num_leaves()) as usize * 2;
		let left = self.children[offset];
		let right = self.children[offset + 1];

		(Node(left), Node(right))
	}

	pub fn other_child(
		&self,
		parent: Internal,
		child: Node,
	) -> Result<Node> {
		let (left, right) = self.children_of(parent);
		if child == left {
			Ok(right)
		} else if child == right {
			Ok(left)
		} else {
			py_bail!(
				PyValueError,
				"Node {child:?} is not a child of {parent:?}",
			);
		}
	}

	/// Index of the edge between `child` and its parent.
	pub fn edge_index(&self, child: Node) -> u32 {
		child.0
	}

	pub fn edge_length(&self, edge: u32) -> f64 {
		let (child, parent) = self.edge_nodes(edge);

		self.height_of(*parent) - self.height_of(child)
	}

	pub fn edge_nodes(&self, edge: u32) -> (Node, Internal) {
		let child = edge;
		let parent = self.parents[child as usize];

		(Node(child), Internal(parent))
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		if self.parents[node.usize()] != TOMBSTONE {
			Some(Internal(self.parents[node.usize()]))
		} else {
			None
		}
	}

	pub fn depth(&self, mut node: Node) -> u32 {
		let mut out = 0;
		while let Some(parent) = self.parent_of(node) {
			out += 1;
			node = *parent;
		}
		out
	}

	pub fn mrca(&self, a: Node, b: Node) -> Internal {
		let out = Tree::mrca(self, a, b);
		self.as_internal(out).expect("unreachable")
	}

	pub fn is_grandparent(&self, node: Internal) -> bool {
		let (left, right) = self.children_of(node);
		self.is_internal(left) && self.is_internal(right)
	}

	pub fn num_grandparents(&self) -> usize {
		let mut out = 0;
		for internal in self.internals() {
			out += usize::from(self.is_grandparent(internal));
		}
		out
	}

	pub fn random_node(&self, rng: &mut Rng) -> Node {
		let i = rng.random_range(0..self.num_nodes());
		Node(i)
	}

	pub fn random_nonroot_node(&self, rng: &mut Rng) -> (Node, Internal) {
		loop {
			let node = self.random_node(rng);
			if let Some(parent) = self.parent_of(node) {
				return (node, parent);
			}
		}
	}

	pub fn random_internal(&self, rng: &mut Rng) -> Internal {
		let i = rng.random_range(self.num_leaves()..self.num_nodes());
		Internal(i)
	}

	pub fn random_nonroot_internal(
		&self,
		rng: &mut Rng,
	) -> (Internal, Internal) {
		loop {
			let node = self.random_internal(rng);
			if let Some(parent) = self.parent_of(*node) {
				return (node, parent);
			}
		}
	}

	pub fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		for (i, leaf_name) in self.names.iter().enumerate() {
			if name == leaf_name {
				return Some(Leaf(i as u32));
			}
		}

		None
	}

	pub fn nodes(&self) -> impl Iterator<Item = Node> + use<> {
		(0..self.num_nodes()).map(Node)
	}

	pub fn internals(&self) -> impl Iterator<Item = Internal> + use<> {
		(self.num_leaves()..self.num_nodes()).map(Internal)
	}

	pub fn leaves(&self) -> impl Iterator<Item = Leaf> + use<> {
		(0..self.num_leaves()).map(Leaf)
	}

	pub fn edges(&self) -> impl Iterator<Item = u32> + use<> {
		let root = self.root().0;
		(0..root).chain(root + 1..self.num_nodes())
	}

	pub fn to_binary(
		&self,
		internal_ids: bool,
		clock_rate: &dyn Fn(u32) -> f64,
	) -> BinaryTree {
		let num_leaves = self.num_leaves();
		let num_nodes = self.num_nodes();
		let num_internals = self.num_internals();

		let mut edge_lengths = Vec::with_capacity(num_nodes as usize);
		for node in self.nodes() {
			let length = if node == *self.root() {
				0.0
			} else {
				self.edge_length(node.0) * clock_rate(node.0)
			};
			edge_lengths.push(length);
		}

		let mut internal_names = ArrayUtf8::<Nullable>::new();
		if internal_ids {
			for internal in self.internals() {
				internal_names
					.push_some(&internal.0.to_string())
					.unwrap();
			}
		} else {
			internal_names.push_nulls(num_internals as usize);
		}

		BinaryTree::new(
			num_leaves,
			self.root().0,
			Buffer::from_slice(&self.children),
			Buffer::from_slice(&self.parents),
			Buffer::from_slice(&edge_lengths),
			self.names.clone(),
			internal_names,
		)
	}

	fn internal_heights(&self) -> Vec<f64> {
		let mut out = Vec::new();
		for internal in self.internals() {
			out.push(self.height_of(*internal));
		}
		out.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
		out
	}
}

impl Tree for DatedTree {
	fn root(&self) -> Option<Node> {
		Some(*self.root())
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
		if let Some(leaf) = self.as_leaf(node) {
			self.names.get(leaf.usize())
		} else {
			""
		}
	}

	fn edge_length(&self, node: Node) -> Option<f64> {
		Some(self.edge_length(node.0))
	}
}

impl Parameter for DatedTree {
	fn is_changed(&self) -> bool {
		self.updated_edges.is_any_on()
	}

	fn accept(&mut self) {
		self.children.accept();
		self.parents.accept();
		self.heights.accept();
		self.clear_updated();
		self.root.1 = self.root.0;
	}

	fn reject(&mut self) {
		self.children.reject();
		self.parents.reject();
		self.heights.reject();
		self.clear_updated();
		self.root.0 = self.root.1;
	}

	fn load(&mut self, bytes: &mut &[u8]) -> Result<()> {
		for i in 0..self.num_nodes() {
			self.heights[i as usize] =
				verbatim::read_f64_le(bytes)?;
		}

		// overwrite all parents as one of them will be left as root
		for internal in self.internals() {
			self.parents[internal.usize()] = TOMBSTONE;
		}

		let num_leaves = self.num_leaves();
		for i in 0..self.num_edges() {
			let child = verbatim::read_u32_le(bytes)?;
			self.children[i as usize] = child;
			let parent = i / 2 + num_leaves;
			self.parents[child as usize] = parent;
		}

		for internal in self.internals() {
			if self.parents[internal.usize()] == TOMBSTONE {
				self.set_root(internal);
			}
		}

		self.mark_all_edges_updated();
		Ok(())
	}

	fn dump(&self, writer: &mut dyn Write) -> Result<()> {
		for height in self.heights.iter().copied() {
			verbatim::write_f64_le(writer, height)?;
		}
		for child in self.children.iter().copied() {
			verbatim::write_u32_le(writer, child)?;
		}

		Ok(())
	}
}

macro_rules! make_iterator {
	($name: ident, $t: tt) => {
		#[pyclass(frozen, module = "aspartik.b3.tree")]
		struct $name {
			current: MonotonicU32,
			end: u32,
		}

		impl $name {
			fn new(start: u32, end: u32) -> Self {
				Self {
					current: start.into(),
					end,
				}
			}
		}

		#[pymethods]
		impl $name {
			fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
				this
			}

			fn __next__(&self) -> Option<$t> {
				let out = self.current.load();
				if out == self.end {
					return None;
				}
				self.current.add(1);
				Some($t(out))
			}
		}
	};
}

make_iterator!(InternalsIter, Internal);
make_iterator!(LeavesIter, Leaf);

#[pyclass(frozen, module = "aspartik.b3.tree")]
struct NodesIter {
	current: MonotonicU32,
	end: u32,
	num_leaves: u32,
}

#[pymethods]
impl NodesIter {
	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__<'py>(
		&self,
		py: Python<'py>,
	) -> Result<Option<Bound<'py, PyAny>>> {
		let current = self.current.load();
		if current == self.end {
			return Ok(None);
		}
		let out = Node(current);

		self.current.add(1);

		out.into_pyobject(py, self.num_leaves).map(Some)
	}
}

#[derive(Debug)]
#[pyclass(name = "DatedTree", module = "aspartik.b3", frozen)]
/// A bifuracting dated phylogenetic tree which supports epoch-versioning
///
/// - "Dated" means that in this tree each node has a certain height and edge
///   lengths are derived from these heights.  Thus, it is impossible to change
///   just the edge length, node heights must be changed instead.
///
/// - Epoch-versioning means that the changes made to the tree can be rolled
///   back or accepted via the `accept` and `reject` methods.
pub struct PyDatedTree {
	inner: Mutex<DatedTree>,
}

impl_pyparameter_common!(PyDatedTree, DatedTree, {
	#[new]
	fn new(names: PyTaxonSet, rng: Py<PyRng>) -> Result<Self> {
		let tree = DatedTree::new(names.0, &mut rng.get().inner())?;
		let tree = Self {
			inner: Mutex::new(tree),
		};
		Ok(tree)
	}

	/// Randomizes the tree structure
	///
	/// This methods creates a random [Prüfer sequence][wiki] and
	/// rearranges the graph according to it.  Note that it will always the
	/// internal node with the largest index (`num_nodes - 1`) the root.
	///
	/// [wiki]: https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence
	fn set_random_edges(&self, rng: Py<PyRng>) {
		self.inner().set_random_edges(&mut rng.get().inner());
	}

	/// Randomizes the heights of internal nodes
	///
	/// Each internal node gets a height distributed uniformly between
	/// `diff` and `2 * diff` plus the height of the highest of its
	/// children.
	fn set_random_heights(&self, diff: f64, rng: Py<PyRng>) {
		self.inner()
			.set_random_heights(diff, &mut rng.get().inner());
	}

	fn load_newick(&self, newick: &PyTreeBuilder) -> Result<()> {
		self.inner().load_newick(&newick.inner())
	}

	/// A list of all leaf names.
	///
	/// The order is the same as leaf indices: the first name is that of
	/// `Leaf(0)`, the second one is `Leaf(1)`, and so on.
	#[getter]
	fn names(&self) -> PyTaxonSet {
		PyTaxonSet(self.inner().names().clone())
	}

	/// Multiplies the heights of all internal nodes by `scale`
	///
	/// ### Exceptions
	///
	/// Throws a `RuntimeError` if any of the internal nodes would be moved
	/// below either of its children.
	fn scale(&self, scale: f64) -> Result<u32> {
		self.inner().scale(scale)
	}

	fn replace_child(
		&self,
		parent: Internal,
		old_child: Node,
		new_child: Node,
	) -> Result<()> {
		self.inner().replace_child(parent, old_child, new_child)
	}

	fn spr(&self, node: Node, other: Node) -> Result<()> {
		self.inner().spr(node, other)
	}

	/// Sets the height of `node` to `height`
	fn set_height(&self, node: Node, height: f64) -> Result<()> {
		self.inner().set_height(node, height);
		Ok(())
	}

	/// Makes `node` the root of the tree
	///
	/// As the topology can be temporarily broken while the edges are being
	/// swapped, `DatedTree` can't automatically figure out which node is the
	/// root one.  So, operators which change the root of the tree have to
	/// update it manually.
	fn set_root(&self, node: Internal) -> Result<()> {
		self.inner().set_root(node);
		Ok(())
	}

	/// Swaps the parents of nodes `a` and `b`
	///
	/// `a` and `b` must not be a descendant/ancestors and neither of them
	/// can be a root node.  If `a` and `b` share the same parent, they
	/// switch polarity (left child becomes the right child and visa
	/// versa).
	fn swap_parents(&self, a: Node, b: Node) -> Result<()> {
		self.inner().swap_parents(a, b)
	}

	/// Total number of nodes in the tree
	#[getter]
	pub fn num_nodes(&self) -> u32 {
		self.inner().num_nodes()
	}

	/// Number of internal nodes (those with children)
	#[getter]
	pub fn num_internals(&self) -> u32 {
		self.inner().num_internals()
	}

	/// Number of leaf nodes
	#[getter]
	fn num_leaves(&self) -> u32 {
		self.inner().num_leaves()
	}

	/// Total number of edges
	#[getter]
	pub fn num_edges(&self) -> u32 {
		self.inner().num_edges()
	}

	/// Returns `True` if the node is internal
	fn is_internal(&self, node: Node) -> Result<bool> {
		Ok(self.inner().is_internal(node))
	}

	/// Returns `True` if the node is a leaf
	fn is_leaf(&self, node: Node) -> Result<bool> {
		Ok(self.inner().is_leaf(node))
	}

	/// Returns the root node of the tree
	///
	/// Note that the root node might change after tree has been edited, so
	/// the returned node is only guaranteed to be root as long as the tree
	/// hasn't been edited.
	#[getter]
	pub fn root(&self) -> Internal {
		self.inner().root()
	}

	/// Returns the height of `node`
	///
	/// Height here means node's age in some unlabeled units.
	fn height_of(&self, node: Node) -> Result<f64> {
		Ok(self.inner().height_of(node))
	}

	/// Returns a tuple of the left and right children of `node`
	///
	/// This function takes the `Internal` type as its input, so it is
	/// guaranteed to always return the children.  See `as_internal` for
	/// converting general nodes to internal ones.
	fn children_of<'py>(
		&self,
		py: Python<'py>,
		node: Internal,
	) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
		let (left, right) = self.inner().children_of(node);

		let (left, right) = (
			left.into_pyobject(py, self.num_leaves())?,
			right.into_pyobject(py, self.num_leaves())?,
		);

		Ok((left, right))
	}

	/// Returns the child of `parent` other than `child`
	///
	/// Throws an error if `child` isn't a child of `parent`.
	fn other_child<'py>(
		&self,
		py: Python<'py>,
		parent: Internal,
		child: Node,
	) -> Result<Bound<'py, PyAny>> {
		let inner = self.inner();
		inner.other_child(parent, child)
			.and_then(|n| n.into_pyobject(py, inner.num_leaves()))
	}

	/// Returns the index of an edge from `child` to its parent
	fn edge_index(&self, child: Node) -> u32 {
		self.inner().edge_index(child)
	}

	/// Returns the length of `edge`
	///
	/// The length is the distance between the parent and the child nodes
	/// of that edge
	fn edge_length(&self, edge: u32) -> f64 {
		self.inner().edge_length(edge)
	}

	/// Returns the `(child, parent)` tuple corresponding to an edge
	fn edge_nodes<'py>(
		&self,
		py: Python<'py>,
		edge: u32,
	) -> Result<(Bound<'py, PyAny>, Internal)> {
		let inner = self.inner();
		let (node, internal) = inner.edge_nodes(edge);
		let node = node.into_pyobject(py, inner.num_leaves())?;

		Ok((node, internal))
	}

	/// Returns the parent of `node`, or `None` for the root node
	fn parent_of(&self, node: Node) -> Result<Option<Internal>> {
		Ok(self.inner().parent_of(node))
	}

	fn depth(&self, node: Node) -> u32 {
		self.inner().depth(node)
	}

	/// The most recent common ancestor of `a` and `b`
	fn mrca(&self, a: Node, b: Node) -> Internal {
		self.inner().mrca(a, b)
	}

	/// Returns `True` if both children of this node are also internal
	fn is_grandparent(&self, node: Internal) -> bool {
		self.inner().is_grandparent(node)
	}

	/// Number of nodes for whom `is_grandparent` returns `True`
	fn num_grandparents(&self) -> usize {
		self.inner().num_grandparents()
	}

	/// An iterator over all of trees nodes
	///
	/// All of the `Leaf` nodes go before `Internal` ones.
	fn nodes(&self) -> NodesIter {
		NodesIter {
			current: 0.into(),
			end: self.num_nodes(),
			num_leaves: self.num_leaves(),
		}
	}

	/// Returns an iterator over internal nodes.
	fn internals(&self) -> InternalsIter {
		let inner = self.inner();
		InternalsIter::new(inner.num_leaves(), inner.num_nodes())
	}

	/// Returns an iterator over all leaves.
	fn leaves(&self) -> LeavesIter {
		let inner = self.inner();
		LeavesIter::new(0, inner.num_leaves())
	}

	/// Samples a random node from the tree.
	fn random_node<'py>(
		&self,
		py: Python<'py>,
		rng: &PyRng,
	) -> Result<Bound<'py, PyAny>> {
		let node = self.inner().random_node(&mut rng.inner());
		node.into_pyobject(py, self.num_leaves())
	}

	/// Samples a random non-root node
	fn random_nonroot_node(
		&self,
		py: Python,
		rng: &PyRng,
	) -> Result<(Py<PyAny>, Internal)> {
		let (node, parent) =
			self.inner().random_nonroot_node(&mut rng.inner());

		let node = node.into_pyobject(py, self.num_leaves())?;

		Ok((node.unbind(), parent))
	}

	/// Samples a random internal node
	fn random_internal(&self, rng: &PyRng) -> Internal {
		self.inner().random_internal(&mut rng.inner())
	}

	/// Samples a random non-root internal node
	fn random_nonroot_internal(&self, rng: &PyRng) -> (Internal, Internal) {
		self.inner().random_nonroot_internal(&mut rng.inner())
	}

	/// Gets a named leaf or `None` if the name is not found
	fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		self.inner().leaf_by_name(name)
	}

	/// The total length of all tree edges
	fn total_length(&self) -> f64 {
		self.inner().total_length()
	}

	/// Returns true if any of the leaves have a non-0 height
	fn has_dated_tips(&self) -> bool {
		self.inner().has_dated_tips()
	}

	/// Throws an exception if a tree is malformed
	///
	/// This function ensures that:
	///
	/// - No leaf has become anyone's parent.
	/// - All parent nodes are older than their children.
	/// - Parents match their children (mismatches can happen when
	///   `update_edge` is used incorrectly).
	/// - There's only one root (two or more can be set with `set_root`).
	/// - The tree is a tree, meaning that topologically it has no cycles
	///   and is connected.
	fn validate(&self) -> Result<()> {
		self.inner().validate()
	}

	#[pyo3(signature = (clock = None))]
	fn to_binary(&self, clock: Option<Py<PyClock>>) -> PyBinaryTree {
		if let Some(clock) = clock {
			let clock = clock.get().inner();
			PyBinaryTree::from(
				self.inner().to_binary(false, &|edge| {
					clock.get_rate(edge)
				}),
			)
		} else {
			PyBinaryTree::from(
				self.inner().to_binary(false, &|_| 1.0),
			)
		}
	}

	fn ola(&self) -> Vec<i32> {
		self.inner().ola()
	}

	/// Sorted heights of internal nodes
	///
	/// This method is inefficient and should only be used for post-run
	/// analysis.
	fn internal_heights(&self) -> Vec<f64> {
		self.inner().internal_heights()
	}
});
