use anyhow::{Result, bail, ensure};
use bytemuck::{Pod, Zeroable, allocation::cast_vec};
use parking_lot::Mutex;
use pyo3::{
	exceptions::{PyTypeError, PyValueError},
	prelude::*,
	types::{PyAny, PyType},
};
use rand::{RngExt, seq::SliceRandom};

use std::{
	cmp::{Reverse, max, min},
	collections::{BinaryHeap, HashMap, VecDeque},
	io::Write,
	mem,
	ops::Deref,
	sync::atomic::{AtomicUsize, Ordering},
};

use super::Parameter;
use crate::impl_pyparameter_common;
use bitmap::Bitmap;
use data::newick::{
	Edge as NewickEdge, Node as NewickNode, NodeIdx as NewickNodeIndex,
	Tree as NewickTree, python::PyTree as PyNewickTree,
};
use rng::{PyRng, Rng};
use sk::{SkBuf, skbuf};
use util::py_bail;
use verbatim::{Deserialize, Serialize};

const ROOT: usize = 0x524f4f54;

#[derive(Debug, Clone)]
pub struct Tree {
	names: Vec<String>,

	/// `(value, backup)`
	root: (usize, usize),

	children: SkBuf<usize>,
	parents: SkBuf<usize>,
	heights: SkBuf<f64>,

	updated_edges: Bitmap,
	/// An array of length num_nodes, where `true` means that the node has
	/// been updated.
	updated_nodes: Bitmap,
}

#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[repr(transparent)]
pub struct Node(usize);

impl Node {
	fn into_pyobject(
		self,
		py: Python,
		num_leaves: usize,
	) -> Result<Bound<PyAny>> {
		let num_nodes = num_leaves * 2 - 1;
		let any = if self.0 < num_leaves {
			Leaf(self.0).into_pyobject(py)?.into_any()
		} else if self.0 < num_nodes {
			Internal(self.0).into_pyobject(py)?.into_any()
		} else {
			unreachable!()
		};
		Ok(any)
	}
}

impl<'py> FromPyObject<'_, 'py> for Node {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Node> {
		if let Ok(internal) = obj.cast::<Internal>() {
			Ok(Node(internal.get().0))
		} else if let Ok(leaf) = obj.cast::<Leaf>() {
			Ok(Node(leaf.get().0))
		} else {
			py_bail!(
				PyTypeError,
				"Expected `Leaf` or `Internal`, got {}",
				obj.get_type().name()?
			);
		}
	}
}

/// Internal anonymous node of the phylogenetic tree.
///
/// Internals are the unnamed nodes which represent most recent common ancestors
/// of leaves and other internals.
#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[pyclass(from_py_object, module = "aspartik.b3.tree", frozen, eq, hash)]
#[repr(transparent)]
pub struct Internal(usize);

/// Leaf node of the phylogenetic tree
///
/// Every leaf node is associated with a concrete sequence.
#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[pyclass(from_py_object, module = "aspartik.b3.tree", frozen, eq, hash)]
#[repr(transparent)]
pub struct Leaf(usize);

macro_rules! nodes_2 {
	($type:ty) => {
		impl $type {
			pub fn index(&self) -> usize {
				self.0
			}
		}

		#[pymethods]
		impl $type {
			fn __repr__(&self) -> String {
				format!("{}({})", stringify!($type), self.0)
			}
		}

		impl Deref for $type {
			type Target = Node;

			fn deref<'a>(&'a self) -> &'a Node {
				// SAFETY: `Leaf`, `Internal`, and `Node` all
				// have the same layout with: `usize` with
				// `repr(transparent)`.  So, casting between
				// them while preserving the lifetimes is safe.
				unsafe {
					mem::transmute::<&'a $type, &'a Node>(
						self,
					)
				}
			}
		}
	};
}

nodes_2!(Leaf);
nodes_2!(Internal);

impl Tree {
	pub fn new(names: Vec<String>, rng: &mut Rng) -> Result<Self> {
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

			root: (0, 0),

			children: skbuf![ROOT; num_internals * 2],
			parents: skbuf![ROOT; num_nodes],
			heights: skbuf![0.0; num_nodes],

			updated_edges: Bitmap::new(num_nodes),
			updated_nodes: Bitmap::new(num_nodes),
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
		let mut prüfer: Vec<usize> =
			internals.clone().chain(internals).collect();
		let root = prüfer.pop().unwrap(); // remove the last node
		prüfer.shuffle(rng); // random shuffle

		let mut parents = vec![ROOT; num_nodes];
		let mut children = vec![ROOT; 2 * num_internals];

		let mut histogram = vec![2; num_internals];
		// the last node only appears once
		*histogram.last_mut().unwrap() = 1;
		let mut unused =
			BinaryHeap::from_iter((0..num_leaves).map(Reverse));

		for parent in prüfer {
			let child = unused.pop().unwrap().0;

			parents[child] = parent;

			// `children` update
			let idx = (parent - num_leaves) * 2;
			// first encountered child goes in the left slot, second
			// one goes in the right
			if children[idx] == ROOT {
				children[idx] = child;
			} else {
				children[idx + 1] = child;
			}

			histogram[parent - num_leaves] -= 1;
			if histogram[parent - num_leaves] == 0 {
				unused.push(Reverse(parent));
			}
		}
		// last node, which should be connected to the root
		let child = unused.pop().unwrap().0;
		parents[child] = root;
		children[(root - num_leaves) * 2 + 1] = child;

		self.parents = parents.into();
		self.children = children.into();
		self.root = (root, root);
	}

	// Sets the heights of internal nodes by walking upwards breadth-first
	// starting with all of the leaves
	pub fn set_random_heights(&mut self, diff: f64, rng: &mut Rng) {
		let mut walk = VecDeque::new();
		for leaf in self.leaves() {
			// All leaves have a parent
			let parent = self.parent_of(*leaf).unwrap();
			walk.push_back(parent);
		}
		while let Some(internal) = walk.pop_front() {
			let (left, right) = self.children_of(internal);
			let max = f64::max(
				self.height_of(left),
				self.height_of(right),
			);

			let node_diff = diff * (1.0 + rng.random::<f64>());
			self.set_height(*internal, max + node_diff);

			if let Some(parent) = self.parent_of(*internal) {
				walk.push_front(parent);
			}
		}

		self.accept();
	}

	pub fn load_newick(&mut self, newick: &NewickTree) -> Result<()> {
		ensure!(newick.leaves().count() == self.num_leaves());
		ensure!(newick.num_nodes() == self.num_nodes());

		let mut mapping = HashMap::<NewickNodeIndex, usize>::new();
		for n_leaf_idx in newick.leaves() {
			let name = newick.get_node(n_leaf_idx).name();
			let Some(s_idx) =
				self.names().iter().position(|n| *n == name)
			else {
				bail!("Node {name} not found in tree");
			};
			mapping.insert(n_leaf_idx, s_idx);
		}

		for i in 0..self.num_edges() {
			self.children.set(i, ROOT);
		}

		let mut queue = VecDeque::from_iter(newick.leaves());
		let num_leaves = self.num_leaves();
		let mut internal_idx = self.num_leaves();
		while let Some(n_idx) = queue.pop_front() {
			let Some(parent) = newick.parent_of(n_idx) else {
				self.set_root(Internal(mapping[&n_idx]));
				continue;
			};
			let s_parent_idx =
				*mapping.entry(parent).or_insert_with(|| {
					queue.push_back(parent);
					let idx = internal_idx;
					internal_idx += 1;
					idx
				});

			let current_idx = mapping[&n_idx];
			self.parents.set(current_idx, s_parent_idx);
			let child_offset = (s_parent_idx - num_leaves) * 2;
			if self.children[child_offset] == ROOT {
				self.children.set(child_offset, current_idx);
			} else {
				self.children
					.set(child_offset + 1, current_idx);
			}
		}

		let rmapping: HashMap<Node, NewickNodeIndex> = mapping
			.into_iter()
			.map(|(k, v)| (Node(v), k))
			.collect();

		self.set_height(*self.root(), 0.0);
		let mut queue = VecDeque::from([*self.root()]);
		while let Some(node) = queue.pop_front() {
			if let Some(node) = self.as_internal(node) {
				let (left, right) = self.children_of(node);
				queue.push_back(left);
				queue.push_back(right);
			}

			let Some(parent) = self.parent_of(node) else {
				continue;
			};
			let parent_height = self.height_of(*parent);
			let edge =
				newick.edge_to_parent(rmapping[&node]).unwrap();
			let Some(edge_length) = edge.distance() else {
				bail!("Encountered Newick node without length");
			};

			self.set_height(node, parent_height - edge_length);
		}
		let mut min = 0.0;
		for &height in self.heights.iter() {
			if height < min {
				min = height;
			}
		}

		for node in self.nodes() {
			self.set_height(node, self.height_of(node) - min);
		}

		// set root
		for internal in self.internals() {
			if self.parents[internal.0] == ROOT {
				self.set_root(internal);
			}
		}

		Ok(())
	}

	pub fn simulate_coalescent(
		names: Vec<String>,
		heights: Vec<f64>,
		population_size: f64,
		rng: &mut Rng,
	) -> Result<Self> {
		ensure!(names.len() == heights.len());
		ensure!(heights.iter().all(|&h| h >= 0.0 && !h.is_nan()));

		let mut tree = Self::new(names.clone(), rng)?;
		for internal in tree.internals() {
			tree.parents.set(internal.0, ROOT);
		}

		for (leaf, &height) in tree.leaves().zip(&heights) {
			tree.set_height(*leaf, height)
		}

		let mut nodes: Vec<_> =
			heights.into_iter().enumerate().collect();

		nodes.sort_unstable_by_key(|(_, height)| height.to_bits());

		let mut current_lineages: Vec<usize> = Vec::new();
		let mut sorted_tips =
			VecDeque::from_iter(nodes.iter().map(|(idx, _)| *idx));
		let mut current_height = 0.0;

		while sorted_tips.len() + current_lineages.len() > 1 {
			// move tips below current heights to the pool
			while let Some(&next_tip) = sorted_tips.front()
				&& nodes[next_tip].1 <= current_height
			{
				current_lineages
					.push(sorted_tips.pop_front().unwrap());
			}

			if current_lineages.len() < 2 {
				if let Some(&next_tip) = sorted_tips.front() {
					current_height = nodes[next_tip].1;
					continue;
				} else {
					bail!("Failed to merge lineages");
				}
			}

			// TODO: proper demographic functions
			let k = current_lineages.len();
			let rate =
				(k * (k - 1)) as f64 / (2.0 * population_size);
			let interval = -rng.random::<f64>().ln() / rate;
			let next_event_height = current_height + interval;

			if let Some(&next_tip) = sorted_tips.front()
				&& next_event_height > nodes[next_tip].1
			{
				// there's a sample ahead, loopback to tip move
				current_height = nodes[next_tip].1;
				continue;
			}

			current_height = next_event_height;

			let i = rng.random_range(0..current_lineages.len());
			let child_left = current_lineages.swap_remove(i);
			let i = rng.random_range(0..current_lineages.len());
			let child_right = current_lineages.swap_remove(i);

			let new_idx = nodes.len();
			nodes.push((new_idx, current_height));

			tree.heights.set(new_idx, current_height);
			let offset = (new_idx - tree.num_leaves()) * 2;
			tree.children.set(offset, child_left);
			tree.children.set(offset + 1, child_right);
			tree.parents.set(child_left, new_idx);
			tree.parents.set(child_right, new_idx);

			current_lineages.push(new_idx);
		}

		tree.accept();
		Ok(tree)
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
		labels.resize(num_nodes, 0);

		let mut clade_founder = vec![0; num_nodes];

		for node in self.postorder() {
			if let Some(leaf) = self.as_leaf(node) {
				clade_founder[leaf.0] = leaf.0
			} else if let Some(internal) = self.as_internal(node) {
				let (left, right) = self.children_of(internal);
				clade_founder[internal.0] = min(
					clade_founder[left.0],
					clade_founder[right.0],
				);
			} else {
				unreachable!()
			}
		}

		let mut clade_splitter = vec![0usize; num_nodes];
		let mut splitter_to_node = vec![0; num_leaves];
		for node in self.preorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};

			let (left, right) = self.children_of(internal);

			let splitter = max(
				clade_founder[left.0],
				clade_founder[right.0],
			);
			clade_splitter[node.0] = splitter;
			labels[node.0] = -(splitter as i32);
			splitter_to_node[splitter] = node.0;
		}

		let mut ola = Vec::new();
		let mut forward_to = Vec::from_iter(0..num_nodes);

		for label in (1..num_leaves).rev() {
			let splitter_node = Internal(splitter_to_node[label]);
			let (left, right) = self.children_of(splitter_node);

			let sibling = if clade_founder[left.0] == label {
				right
			} else {
				left
			};

			let mut curr = sibling;
			while forward_to[curr.0] != curr.0 {
				curr = Node(forward_to[curr.0]);
			}

			ola.push(labels[curr.0]);
			forward_to[splitter_node.0] = curr.0;
		}
		ola.reverse();

		ola
	}

	pub fn names(&self) -> &[String] {
		&self.names
	}

	pub fn scale(&mut self, scale: f64) -> Result<usize> {
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
		self.updated_nodes.set_all_off();
	}

	pub fn mark_edge_updated(&mut self, edge: usize) {
		self.updated_edges.set_on(edge);
	}

	/// Mark all edges as requiring updates
	///
	/// Also marks all nodes as updated.  The latter is needed because the
	/// clock and the substituion model might call this method via
	/// `Transitions`.  And it might make sense to the requirement to mark
	/// nodes in the future, instead computing them from edges.
	pub fn mark_all_edges_updated(&mut self) {
		self.updated_edges.set_all_on();
		self.updated_nodes.set_all_on();
	}

	pub fn edges_to_update(&self) -> Vec<usize> {
		let mut out = Vec::new();
		for edge in self.edges() {
			if self.is_edge_updated(edge) {
				out.push(edge);
			}
		}
		out
	}

	/// For each updated node go upwards in the tree until root and
	/// mark nodes as updated
	fn mark_updated_parents(&mut self) {
		for node in self.nodes() {
			if self.is_node_updated(node) {
				let mut curr = node;
				while let Some(parent) = self.parent_of(curr) {
					// return early when we find an already
					// visited node to avoid wasting time
					// on already checked paths
					if self.is_node_updated(*parent) {
						break;
					}
					self.mark_node_updated(*parent);
					curr = *parent;
				}
			}
		}
	}

	fn updated_internals(&self) -> Vec<Internal> {
		let mut internals = Vec::from([self.root()]);
		let mut queue = VecDeque::from([self.root()]);

		while let Some(node) = queue.pop_front() {
			let (left, right) = self.children_of(node);

			if self.is_node_updated(left)
				&& let Some(left) = self.as_internal(left)
			{
				internals.push(left);
				queue.push_back(left);
			}
			if self.is_node_updated(right)
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

	pub fn propagation_lists(
		&mut self,
	) -> (Vec<usize>, Vec<[usize; 2]>, usize) {
		// mark updated nodes whose parent edge got updated
		for edge in self.edges() {
			if self.is_edge_updated(edge) {
				let (child, _) = self.edge_nodes(edge);
				self.mark_node_updated(child);
			}
		}
		self.mark_updated_parents();

		let mut nodes = Vec::<Node>::new();

		// Updated leaves, in order
		for leaf in self.leaves() {
			if self.is_node_updated(*leaf) {
				nodes.push(*leaf);
			}
		}
		let num_updated_leaves = nodes.len();

		let internals = self.updated_internals();
		let children = self.children_list(&internals);

		nodes.append(&mut cast_vec(internals));

		(cast_vec(nodes), cast_vec(children), num_updated_leaves)
	}

	pub fn partials_lists(&mut self) -> (Vec<usize>, Vec<[usize; 2]>) {
		// mark nodes whose child edges got updated
		for edge in self.edges() {
			if self.is_edge_updated(edge) {
				let (_, parent) = self.edge_nodes(edge);
				self.mark_node_updated(*parent);
			}
		}
		self.mark_updated_parents();

		let internals = self.updated_internals();
		let children = self.children_list(&internals);

		(cast_vec(internals), cast_vec(children))
	}

	fn mark_node_updated(&mut self, node: Node) {
		self.updated_nodes.set_on(node.0);
	}

	fn is_node_updated(&self, node: Node) -> bool {
		self.updated_nodes.at(node.0)
	}

	fn is_edge_updated(&self, edge: usize) -> bool {
		self.updated_edges.at(edge)
	}

	pub fn is_node_height_updated(&self, node: Node) -> bool {
		self.heights.is_changed_at(node.0)
	}

	pub fn replace_child(
		&mut self,
		parent: Internal,
		old_child: Node,
		new_child: Node,
	) -> Result<()> {
		self.parents.set(new_child.0, parent.0);

		let (left, right) = self.children_of(parent);
		let idx = (parent.0 - self.num_leaves()) * 2;

		if old_child == left {
			self.children.set(idx, new_child.0);
		} else if old_child == right {
			self.children.set(idx + 1, new_child.0);
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
		self.heights.set(node.0, height);

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
		self.parents.set(node.0, ROOT);
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

		let roots: Vec<usize> = self
			.parents
			.iter()
			.enumerate()
			.filter(|&(_, &parent)| parent == ROOT)
			.map(|(idx, _)| idx)
			.collect();
		ensure!(
			roots.len() == 1,
			"The tree has more than one root: {:?}",
			roots
		);
		ensure!(roots[0] == self.root.0);

		use std::collections::HashSet;
		let mut children = HashSet::new();
		for node in self.internals() {
			let (left, right) = self.children_of(node);
			children.insert(left);
			children.insert(right);
		}
		ensure!(children.len() == self.num_nodes() - 1);

		Ok(())
	}

	pub fn num_nodes(&self) -> usize {
		self.heights.len()
	}

	pub fn num_internals(&self) -> usize {
		(self.num_nodes() - 1) / 2
	}

	pub fn num_leaves(&self) -> usize {
		self.num_internals() + 1
	}

	pub fn num_edges(&self) -> usize {
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
		self.heights[node.0]
	}

	pub fn children_of(&self, node: Internal) -> (Node, Node) {
		let index = node.0 - self.num_leaves();
		let left = self.children[index * 2];
		let right = self.children[index * 2 + 1];

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
	pub fn edge_index(&self, child: Node) -> usize {
		child.0
	}

	pub fn edge_length(&self, edge: usize) -> f64 {
		let (child, parent) = self.edge_nodes(edge);

		self.height_of(*parent) - self.height_of(child)
	}

	pub fn edge_nodes(&self, edge: usize) -> (Node, Internal) {
		let child = edge;
		let parent = self.parents[child];

		(Node(child), Internal(parent))
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		if self.parents[node.0] != ROOT {
			Some(Internal(self.parents[node.0]))
		} else {
			None
		}
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
				return Some(Leaf(i));
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

	pub fn edges(&self) -> impl Iterator<Item = usize> + use<> {
		let root = self.root().0;
		(0..root).chain(root + 1..self.num_nodes())
	}

	pub fn to_newick(&self, internal_ids: bool) -> String {
		let mut tree = NewickTree::new();

		let mut map = HashMap::<Node, NewickNodeIndex>::new();

		for node in self.nodes() {
			let name = if self.is_leaf(node) {
				self.names[node.0].clone()
			} else if internal_ids {
				node.0.to_string()
			} else {
				String::new()
			};

			let newick_node = tree
				.add_node(NewickNode::new(name, String::new()));

			map.insert(node, newick_node);
		}

		for node in self.nodes() {
			let Some(parent) = self.parent_of(node) else {
				tree.set_root(map[&node]);
				continue;
			};

			let edge_len = self.edge_length(node.0);
			let edge =
				NewickEdge::new(Some(edge_len), String::new());

			tree.add_edge(map[&parent], map[&node], edge);
		}

		tree.into_string()
	}

	fn postorder(&self) -> Postorder<'_> {
		Postorder {
			tree: self,
			stack: vec![(*self.root(), false)],
		}
	}

	fn preorder(&self) -> Preorder<'_> {
		Preorder {
			tree: self,
			stack: vec![*self.root()],
		}
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

struct Postorder<'a> {
	tree: &'a Tree,
	stack: Vec<(Node, bool)>,
}

impl Iterator for Postorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Node> {
		while let Some((node, children_visited)) = self.stack.pop() {
			if children_visited {
				return Some(node);
			}

			self.stack.push((node, true));

			let Some(internal) = self.tree.as_internal(node) else {
				continue;
			};

			let (left, right) = self.tree.children_of(internal);
			self.stack.push((right, false));
			self.stack.push((left, false));
		}
		None
	}
}

struct Preorder<'a> {
	tree: &'a Tree,
	stack: Vec<Node>,
}

impl Iterator for Preorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Node> {
		let node = self.stack.pop()?;

		if let Some(internal) = self.tree.as_internal(node) {
			let (left, right) = self.tree.children_of(internal);
			self.stack.push(right);
			self.stack.push(left);
		}

		Some(node)
	}
}

impl Parameter for Tree {
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
			self.heights.set(i, f64::deserialize(bytes)?);
		}

		// overwrite all parents as one of them will be left as root
		for internal in self.internals() {
			self.parents.set(internal.0, ROOT);
		}

		let num_leaves = self.num_leaves();
		for i in 0..self.num_edges() {
			let child = u32::deserialize(bytes)? as usize;
			self.children.set(i, child);
			let parent = i / 2 + num_leaves;
			self.parents.set(child, parent);
		}

		for internal in self.internals() {
			if self.parents[internal.0] == ROOT {
				self.set_root(internal);
			}
		}

		self.mark_all_edges_updated();
		Ok(())
	}

	fn dump(&self, writer: &mut dyn Write) -> Result<()> {
		for &height in &self.heights {
			height.serialize(writer)?;
		}
		for &child in &self.children {
			(child as u32).serialize(writer)?;
		}

		Ok(())
	}
}

macro_rules! make_iterator {
	($name: ident, $t: tt) => {
		#[pyclass(frozen, module = "aspartik.b3.tree")]
		struct $name {
			current: AtomicUsize,
			end: usize,
		}

		impl $name {
			fn new(start: usize, end: usize) -> Self {
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
				let out = self.current.load(Ordering::Relaxed);
				if out == self.end {
					return None;
				}
				self.current.fetch_add(1, Ordering::Relaxed);
				Some($t(out))
			}
		}
	};
}

make_iterator!(InternalsIter, Internal);
make_iterator!(LeavesIter, Leaf);

#[pyclass(module = "aspartik.b3.tree")]
struct NodesIter {
	current: usize,
	end: usize,
	num_leaves: usize,
}

#[pymethods]
impl NodesIter {
	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__<'py>(
		&mut self,
		py: Python<'py>,
	) -> Result<Option<Bound<'py, PyAny>>> {
		if self.current == self.end {
			return Ok(None);
		}

		let out = Node(self.current);

		self.current += 1;

		Ok(Some(out.into_pyobject(py, self.num_leaves)?))
	}
}

#[derive(Debug)]
#[pyclass(name = "Tree", module = "aspartik.b3", frozen)]
/// A phylogenetic tree
///
/// Unlike BEAST2, where the tree is implemented as a collection of nodes
/// pointing to each other, in `b3` `Tree` is a self-contained data structure
/// which holds all of the topology and heights.  This means that nodes
/// (`Internal` and `Leaf`) are identifiers of nodes in a given `Tree` object,
/// much like indices of an array.  This means that all operations, such as
/// getting parents of node heights, have to go through `Tree`'s methods.
///
/// The current implementation only supports bifurcating trees.
pub struct PyTree {
	inner: Mutex<Tree>,
}

impl_pyparameter_common!(PyTree, Tree, {
	#[new]
	fn new(names: Vec<String>, rng: Py<PyRng>) -> Result<Self> {
		let tree = Tree::new(names, &mut rng.get().inner())?;
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

	fn load_newick(&self, newick: &PyNewickTree) -> Result<()> {
		self.inner().load_newick(&newick.inner())
	}

	#[classmethod]
	fn simulate_coalescent(
		_cls: Py<PyType>,
		names: Vec<String>,
		heights: Vec<f64>,
		population_size: f64,
		rng: Py<PyRng>,
	) -> Result<Self> {
		let tree = Tree::simulate_coalescent(
			names,
			heights,
			population_size,
			&mut rng.get().inner(),
		)?;
		Ok(Self {
			inner: Mutex::new(tree),
		})
	}

	/// A list of all leaf names.
	///
	/// The order is the same as leaf indices: the first name is that of
	/// `Leaf(0)`, the second one is `Leaf(1)`, and so on.
	#[getter]
	fn names(&self) -> Vec<String> {
		self.inner().names().to_vec()
	}

	/// Multiplies the heights of all internal nodes by `scale`
	///
	/// ### Exceptions
	///
	/// Throws a `RuntimeError` if any of the internal nodes would be moved
	/// below either of its children.
	fn scale(&self, scale: f64) -> Result<usize> {
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
	/// swapped, `Tree` can't automatically figure out which node is the
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
	pub fn num_nodes(&self) -> usize {
		self.inner().num_nodes()
	}

	/// Number of internal nodes (those with children)
	#[getter]
	pub fn num_internals(&self) -> usize {
		self.inner().num_internals()
	}

	/// Number of leaf nodes
	#[getter]
	fn num_leaves(&self) -> usize {
		self.inner().num_leaves()
	}

	/// Total number of edges
	#[getter]
	pub fn num_edges(&self) -> usize {
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
	fn edge_index(&self, child: Node) -> usize {
		self.inner().edge_index(child)
	}

	/// Returns the length of `edge`
	///
	/// The length is the distance between the parent and the child nodes
	/// of that edge
	fn edge_length(&self, edge: usize) -> f64 {
		self.inner().edge_length(edge)
	}

	/// Returns the `(child, parent)` tuple corresponding to an edge
	fn edge_nodes<'py>(
		&self,
		py: Python<'py>,
		edge: usize,
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
			current: 0,
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

	/// Returns the tree topology in the Newick format
	///
	/// Leaf nodes will be labeled with the names passed to the constructor
	/// while the internal nodes are unlabeled.
	#[pyo3(signature = (internal_ids = false))]
	fn to_newick(&self, internal_ids: bool) -> String {
		self.inner().to_newick(internal_ids)
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
