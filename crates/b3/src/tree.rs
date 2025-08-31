use anyhow::{Result, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::{
	exceptions::PyTypeError,
	types::{PyAny, PyDict, PyTuple},
};
use rand::Rng as _;
use rand::distr::{Distribution, Uniform};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use std::{
	cmp::Reverse,
	collections::{BinaryHeap, VecDeque},
	mem,
	ops::Deref,
};

use io::newick::{
	Node as NewickNode, NodeIndex as NewickNodeIndex, Tree as NewickTree,
};
use rng::{PyRng, Rng};
use skvec::{SkVec, skvec};
use util::{py_bail, py_pickle_state_impl};

const ROOT: usize = usize::MAX;

#[derive(Debug, Serialize, Deserialize)]
pub struct Tree {
	names: Vec<String>,

	children: SkVec<usize>,
	parents: SkVec<usize>,
	heights: SkVec<f64>,

	updated_edges: Vec<usize>,
	/// An array of length num_nodes, where `true` means that the node has
	/// been updated.
	updated_nodes: Box<[bool]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<'py> FromPyObject<'py> for Node {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Node> {
		if let Ok(internal) = obj.downcast::<Internal>() {
			Ok(Node(internal.get().0))
		} else if let Ok(leaf) = obj.downcast::<Leaf>() {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[pyclass(frozen, module = "aspartik.b3.tree")]
#[repr(transparent)]
pub struct Internal(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[pyclass(frozen, module = "aspartik.b3.tree")]
#[repr(transparent)]
pub struct Leaf(usize);

macro_rules! nodes_2 {
	($type:ty) => {
		#[pymethods]
		impl $type {
			fn __repr__(&self) -> String {
				format!("{}({})", stringify!($type), self.0)
			}

			fn __eq__(&self, other: Node) -> bool {
				self.0 == other.0
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
	pub fn new(names: Vec<String>, rng: &mut Rng) -> Self {
		let num_leaves = names.len();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		// Here we create a Prüfer sequence, which encodes a binary tree
		// with the root in the last node with the ID `2l - 2`.  To do
		// that we create a sequence in which all internal nodes appear
		// twice.  Except the last node, which only appears once.
		let internals = num_leaves..num_nodes;
		let mut prüfer: Vec<usize> =
			internals.clone().chain(internals).collect();
		prüfer.pop(); // remove the last node
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
		let root = num_nodes - 1;
		parents[child] = root;
		children[(root - num_leaves) * 2 + 1] = child;

		Self {
			names,

			children: children.into(),
			parents: parents.into(),
			heights: skvec![0.0; num_nodes],

			updated_edges: Vec::new(),
			updated_nodes: vec![false; num_nodes].into(),
		}
	}

	// Sets the heights of internal nodes by walking upwards breadth-first
	// starting with all of the leaves
	pub fn set_random_heights(&mut self, rng: &mut Rng) {
		const DIFF: f64 = 0.1;

		let mut walk = VecDeque::new();
		for leaf in self.leaves() {
			// All leaves have a parent
			let parent = self.parent_of(&leaf).unwrap();
			walk.push_back(parent);
		}
		while let Some(internal) = walk.pop_front() {
			let (left, right) = self.children_of(&internal);
			let max = f64::max(
				self.height_of(&left),
				self.height_of(&right),
			);

			let diff = DIFF * (2.0 + rng.random::<f64>());
			self.set_height(&internal, max + diff);

			if let Some(parent) = self.parent_of(&internal) {
				walk.push_front(parent);
			}
		}
	}

	pub fn accept(&mut self) {
		self.children.accept();
		self.parents.accept();
		self.heights.accept();
		self.clear_updated();
	}

	pub fn reject(&mut self) {
		self.children.reject();
		self.parents.reject();
		self.heights.reject();
		self.clear_updated();
	}

	fn clear_updated(&mut self) {
		self.updated_edges.clear();
		for node_status in &mut self.updated_nodes {
			*node_status = false;
		}
	}

	pub fn edges_to_update(&self) -> Vec<usize> {
		self.updated_edges.clone()
	}

	pub fn nodes_to_update(&mut self) -> (Vec<Node>, usize) {
		let mut nodes = Vec::<Node>::with_capacity(self.num_nodes());

		// For each updated node go upwards in the tree until root and
		// mark nodes as updated
		for node in (0..self.num_nodes()).map(Node) {
			if self.updated_nodes[node.0] {
				let mut curr = node;
				while let Some(parent) = self.parent_of(&curr) {
					// return early when we find an already
					// visited node to avoid wasting time on
					// already checked paths
					if self.updated_nodes[parent.0] {
						break;
					}
					self.mark_updated(&parent);
					curr = *parent;
				}
			}
		}

		// Updated leaves, in order
		for i in 0..self.num_leaves() {
			if self.updated_nodes[i] {
				nodes.push(Node(i));
			}
		}
		let num_updated_leaves = nodes.len();

		// current rank
		let mut current = Vec::from([self.root()]);
		// next rank
		let mut next = Vec::<Internal>::new();
		// collects nodes for output
		let mut internals = Vec::<Node>::new();

		while !current.is_empty() {
			// For each rank we fill out the next rank with internal
			// nodes.  When there are no more ranks the updated
			// `current` will be empty and the iteration will stop
			for node in current.iter() {
				let (left, right) = self.children_of(node);

				if let Some(left) = self.as_internal(&left)
					&& self.updated_nodes[left.0]
				{
					next.push(left);
				}
				if let Some(right) = self.as_internal(&right)
					&& self.updated_nodes[right.0]
				{
					next.push(right);
				}
			}
			internals.extend(current
				.iter()
				.map(|n| -> Node { *n.deref() }));
			current = std::mem::take(&mut next);
		}

		internals.reverse();
		internals.pop(); // remove root
		nodes.append(&mut internals);

		(nodes, num_updated_leaves)
	}

	/// A breadth-first order of internals starting from the root.
	pub fn full_update(&mut self) -> (Vec<Node>, usize) {
		for edited in &mut self.updated_nodes {
			*edited = true;
		}

		self.nodes_to_update()
	}

	pub fn to_lists(
		&self,
		nodes: &[Node],
	) -> (Vec<usize>, Vec<usize>, usize) {
		let root = self.root().0;

		let num_nodes = nodes.len();
		let mut out_nodes = Vec::with_capacity(num_nodes);
		let mut edges = Vec::with_capacity(num_nodes * 2);

		for node in nodes {
			out_nodes.push(node.0);
			edges.push(self.edge_index(node));
		}

		(out_nodes, edges, root)
	}

	fn mark_updated(&mut self, node: &Node) {
		self.updated_nodes[node.0] = true;
	}

	/// Overwrites the child of `edge` with `new_child`.  Only `edge` and
	/// `new_child` changes are recorded, it is presumed that the operator
	/// will call another method for the old child and `new_child`'s parent
	/// edge.
	fn update_edge(&mut self, edge: usize, new_child: &Node) {
		let (_, parent) = self.edge_nodes(edge);

		self.children.set(edge, new_child.0);
		self.parents.set(new_child.0, parent.0);

		self.updated_edges.push(edge);

		// `parent` is now the parent of `new_child`, so it'll
		// be updated.  The operator must handle the old node
		// separately.
		self.mark_updated(new_child);
	}

	/// Sets the height of `node`, recording it and it's parent and child
	/// edges (if it has those).
	pub fn set_height(&mut self, node: &Node, height: f64) {
		self.heights.set(node.0, height);
		self.mark_updated(node);

		if self.parent_of(node).is_some() {
			self.updated_edges.push(self.edge_index(node));
		}
		if let Some(node) = self.as_internal(node) {
			let (left, right) = self.children_of(&node);
			self.updated_edges.push(self.edge_index(&left));
			self.updated_edges.push(self.edge_index(&right));

			self.mark_updated(&left);
			self.mark_updated(&right);
		}
	}

	/// Doesn't overwrite the old root.
	pub fn set_root(&mut self, node: &Node) {
		self.parents.set(node.0, ROOT);
	}

	/// Replaces `child` with `replacement` in respect to `child`'s parent.
	pub fn update_replacement(&mut self, child: &Node, replacement: &Node) {
		let edge = self.edge_index(child);
		self.update_edge(edge, replacement);
	}

	// TODO: invariants (a can't be a parent of b)
	pub fn swap_parents(&mut self, a: &Node, b: &Node) {
		assert!(self.parent_of(a).is_some(), "a must not be root");
		assert!(self.parent_of(b).is_some(), "b must not be root");

		let edge_a = self.edge_index(a);
		let edge_b = self.edge_index(b);

		self.update_edge(edge_a, b);
		self.update_edge(edge_b, a);
	}

	#[expect(unused)]
	pub(crate) fn node_is_valid(&self, node: &Node) -> bool {
		if let Some(leaf) = self.as_leaf(node) {
			let parent = self.parent_of(&leaf).unwrap();
			self.height_of(&leaf) < self.height_of(&parent)
		} else if let Some(internal) = self.as_internal(node) {
			let (left, right) = self.children_of(&internal);
			if left == right {
				return false;
			}

			let (left_height, right_height) =
				(self.height_of(&left), self.height_of(&right));
			let height = self.height_of(&internal);

			height > left_height && height > right_height
		} else {
			unreachable!()
		}
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
			let (left, right) = self.children_of(&node);

			ensure!(
				self.height_of(&node) > self.height_of(&left),
				"Node {} ({}) is younger than it's left child {} ({})",
				node.0,
				self.height_of(&node),
				left.0,
				self.height_of(&left),
			);
			ensure!(
				self.height_of(&node) > self.height_of(&right),
				"Node {} ({}) is younger than it's right child {} ({})",
				node.0,
				self.height_of(&node),
				right.0,
				self.height_of(&right),
			);

			let left_parent = self.parent_of(&left);
			let right_parent = self.parent_of(&right);
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
			.copied()
			.filter(|p| *p == ROOT)
			.collect();
		ensure!(
			roots.len() == 1,
			"The tree has more than one root: {:?}",
			roots
		);

		use std::collections::HashSet;
		let mut children = HashSet::new();
		for node in self.internals() {
			let (left, right) = self.children_of(&node);
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

	pub fn is_internal(&self, node: &Node) -> bool {
		node.0 >= self.num_leaves()
	}

	pub fn is_leaf(&self, node: &Node) -> bool {
		node.0 < self.num_leaves()
	}

	pub fn as_internal(&self, node: &Node) -> Option<Internal> {
		if self.is_internal(node) {
			Some(Internal(node.0))
		} else {
			None
		}
	}

	pub fn as_leaf(&self, node: &Node) -> Option<Leaf> {
		if self.is_leaf(node) {
			Some(Leaf(node.0))
		} else {
			None
		}
	}

	/// # Panics
	///
	/// Panics if the tree is malformed and has no root.  This can happen
	/// between the calls to `root` and `update_edge`, for example.
	pub fn root(&self) -> Internal {
		// There must always be a rooted element in the tree.
		let i = self.parents.iter().position(|p| *p == ROOT).unwrap();
		Internal(i)
	}

	pub fn height_of(&self, node: &Node) -> f64 {
		self.heights[node.0]
	}

	pub fn children_of(&self, node: &Internal) -> (Node, Node) {
		let index = node.0 - self.num_leaves();
		let left = self.children[index * 2];
		let right = self.children[index * 2 + 1];

		(Node(left), Node(right))
	}

	/// Index of the edge between `child` and its parent.
	///
	/// # Panics
	///
	/// Panics if `child` is root.
	pub fn edge_index(&self, child: &Node) -> usize {
		let parent = self.parent_of(child).unwrap();

		if &self.children_of(&parent).0 == child {
			(parent.0 - self.num_leaves()) * 2
		} else {
			(parent.0 - self.num_leaves()) * 2 + 1
		}
	}

	pub fn edge_distance(&self, edge: usize) -> f64 {
		let (child, parent) = self.edge_nodes(edge);

		self.height_of(&parent) - self.height_of(&child)
	}

	fn edge_nodes(&self, edge: usize) -> (Node, Internal) {
		let parent = edge / 2 + self.num_leaves();
		let child = self.children[edge];

		(Node(child), Internal(parent))
	}

	pub fn parent_of(&self, node: &Node) -> Option<Internal> {
		Some(self.parents[node.0])
			.take_if(|p| *p != ROOT)
			.map(Internal)
	}

	pub fn is_grandparent(&self, node: &Internal) -> bool {
		let (left, right) = self.children_of(node);
		self.is_internal(&left) && self.is_internal(&right)
	}

	pub fn num_grandparents(&self) -> usize {
		let mut out = 0;
		for internal in self.internals() {
			out += usize::from(self.is_grandparent(&internal));
		}
		out
	}

	pub fn random_node(&self, rng: &mut Rng) -> Node {
		let range = Uniform::new(0, self.num_nodes()).unwrap();
		let i = range.sample(rng);
		Node(i)
	}

	pub fn random_internal(&self, rng: &mut Rng) -> Internal {
		let range = Uniform::new(self.num_leaves(), self.num_nodes())
			.unwrap();
		let i = range.sample(rng);
		Internal(i)
	}

	pub fn random_leaf(&self, rng: &mut Rng) -> Leaf {
		let range = Uniform::new(0, self.num_leaves()).unwrap();
		let i = range.sample(rng);
		Leaf(i)
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

	pub fn to_newick(&self) -> String {
		let mut tree = NewickTree::new();

		use std::collections::HashMap;
		let mut map: HashMap<Node, NewickNodeIndex> = HashMap::new();

		for node in self.nodes() {
			let distance;
			if let Some(parent) = self.parent_of(&node) {
				distance = self.height_of(&parent)
					- self.height_of(&node);
			} else {
				distance = 0.0;
			}

			let name = if self.is_leaf(&node) {
				self.names[node.0].clone()
			} else {
				String::new()
			};

			let newick_node = tree.add_node(NewickNode::new(
				name,
				Some(distance),
				String::new(),
			));

			map.insert(node, newick_node);
		}

		for parent in self.internals() {
			let (left, right) = self.children_of(&parent);

			tree.add_edge(map[&parent], map[&left]);
			tree.add_edge(map[&parent], map[&right]);

			// set root
			if self.parent_of(&parent).is_none() {
				tree.set_root(map[&parent]);
			}
		}

		tree.serialize()
	}
}

macro_rules! make_iterator {
	($name: ident, $t: tt) => {
		#[pyclass(frozen, module = "aspartik.b3.tree")]
		struct $name {
			current: Mutex<usize>,
			end: usize,
		}

		impl $name {
			fn new(start: usize, end: usize) -> Self {
				Self {
					current: Mutex::new(start),
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
				let mut current = self.current.lock();
				if *current == self.end {
					return None;
				}

				let out = *current;
				*current += 1;
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
/// A phylogenetic bifurcating tree.
///
/// The leaf nodes are derived from the data samples.  Anonymous internal nodes
/// are created automatically.
pub struct PyTree {
	inner: Mutex<Tree>,
}

impl PyTree {
	pub fn inner(&self) -> MutexGuard<'_, Tree> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyTree {
	#[new]
	fn new(names: Vec<String>, rng: Py<PyRng>) -> Result<Self> {
		let tree = Tree::new(names, &mut rng.get().inner());
		let tree = Self {
			inner: Mutex::new(tree),
		};
		Ok(tree)
	}

	fn set_random_heights(&self, rng: Py<PyRng>) {
		self.inner().set_random_heights(&mut rng.get().inner());
	}

	fn update_edge(&self, edge: usize, new_child: Node) -> Result<()> {
		self.inner().update_edge(edge, &new_child);
		Ok(())
	}

	fn set_height(&self, node: Node, height: f64) -> Result<()> {
		self.inner().set_height(&node, height);
		Ok(())
	}

	fn set_root(&self, node: Node) -> Result<()> {
		self.inner().set_root(&node);
		Ok(())
	}

	fn swap_parents(&self, a: Node, b: Node) -> Result<()> {
		self.inner().swap_parents(&a, &b);
		Ok(())
	}

	#[getter]
	fn num_nodes(&self) -> usize {
		self.inner().num_nodes()
	}

	#[getter]
	fn num_internals(&self) -> usize {
		self.inner().num_internals()
	}

	#[getter]
	fn num_leaves(&self) -> usize {
		self.inner().num_leaves()
	}

	fn is_internal(&self, node: Node) -> Result<bool> {
		Ok(self.inner().is_internal(&node))
	}

	fn is_leaf(&self, node: Node) -> Result<bool> {
		Ok(self.inner().is_leaf(&node))
	}

	fn as_internal(&self, node: Node) -> Option<Internal> {
		self.inner().as_internal(&node)
	}

	fn as_leaf(&self, node: Node) -> Option<Leaf> {
		self.inner().as_leaf(&node)
	}

	fn root(&self) -> Internal {
		self.inner().root()
	}

	fn height_of(&self, node: Node) -> Result<f64> {
		Ok(self.inner().height_of(&node))
	}

	fn children_of<'py>(
		&self,
		py: Python<'py>,
		node: Internal,
	) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
		let (left, right) = self.inner().children_of(&node);

		let (left, right) = (
			left.into_pyobject(py, self.num_leaves())?,
			right.into_pyobject(py, self.num_leaves())?,
		);

		Ok((left, right))
	}

	fn edge_index(&self, child: Node) -> Result<usize> {
		Ok(self.inner().edge_index(&child))
	}

	fn edge_distance(&self, edge: usize) -> f64 {
		self.inner().edge_distance(edge)
	}

	fn parent_of(&self, node: Node) -> Result<Option<Internal>> {
		Ok(self.inner().parent_of(&node))
	}

	fn is_grandparent(&self, node: &Internal) -> bool {
		self.inner().is_grandparent(node)
	}

	fn num_grandparents(&self) -> usize {
		self.inner().num_grandparents()
	}

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

	/// Samples a random internal node from a tree.
	fn random_internal(&self, rng: &PyRng) -> Internal {
		self.inner().random_internal(&mut rng.inner())
	}

	/// Samples a random leaf node from a tree.
	fn random_leaf(&self, rng: &PyRng) -> Leaf {
		self.inner().random_leaf(&mut rng.inner())
	}

	fn validate(&self) -> Result<()> {
		self.inner().validate()
	}

	fn accept(&self) {
		self.inner().accept()
	}

	fn reject(&self) {
		self.inner().reject()
	}

	fn newick(&self) -> String {
		self.inner().to_newick()
	}

	// protocols

	// pickle
	fn __getnewargs__<'py>(
		&self,
		py: Python<'py>,
	) -> PyResult<Bound<'py, PyTuple>> {
		let inner = &*self.inner.lock();
		let dummy_rng = PyRng::new(Some(0))?;

		// the tree will be overwritten by `__setstate__`, so we're
		// passing no names and a dummy RNG
		(inner.names.clone(), dummy_rng).into_pyobject(py)
	}
}

py_pickle_state_impl!(PyTree, _tree_pickle_impl);

pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
	let m = PyModule::new(py, "tree")?;

	m.add_class::<Leaf>()?;
	m.add_class::<Internal>()?;

	let locals = PyDict::new(py);
	locals.set_item("m", &m)?;
	py.run(c"m.Node = m.Leaf | m.Internal", None, Some(&locals))?;

	Ok(m)
}
