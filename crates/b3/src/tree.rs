use anyhow::{ensure, Result};
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::{
	exceptions::PyTypeError,
	types::{PyAny, PyDict},
};
use rand::distr::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng as _;
use serde::{Deserialize, Serialize};

use std::{
	cmp::Reverse,
	collections::{BinaryHeap, HashSet, VecDeque},
};

use io::newick::{
	Node as NewickNode, NodeIndex as NewickNodeIndex, Tree as NewickTree,
};
use rng::{PyRng, Rng};
use skvec::SkVec;
use util::py_bail;

const ROOT: usize = usize::MAX;

#[derive(Debug)]
pub struct Tree {
	names: Vec<String>,

	children: SkVec<usize>,
	parents: SkVec<usize>,
	weights: SkVec<f64>,

	updated_edges: Vec<usize>,
	updated_nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
			let node = *internal.get();
			Ok(node.into())
		} else if let Ok(leaf) = obj.downcast::<Leaf>() {
			let node = *leaf.get();
			Ok(node.into())
		} else {
			py_bail!(
				PyTypeError,
				"Expected `Leaf` or `Internal`, got {}",
				obj.get_type().name()?
			);
		}
	}
}

impl From<Internal> for Node {
	fn from(internal: Internal) -> Node {
		Self(internal.0)
	}
}

impl From<Leaf> for Node {
	fn from(leaf: Leaf) -> Node {
		Node(leaf.0)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(frozen, module = "aspartik.b3.tree")]
pub struct Internal(usize);

#[pymethods]
impl Internal {
	fn __repr__(&self) -> String {
		format!("Internal({})", self.0)
	}

	fn __eq__(&self, other: Bound<PyAny>) -> Result<bool> {
		if let Ok(node) = other.downcast::<Internal>() {
			Ok(self.0 == node.get().0)
		} else if other.downcast::<Leaf>().is_ok() {
			Ok(false)
		} else {
			py_bail!(
				PyTypeError,
				"Expected `Leaf` or `Internal`, got {}",
				other.get_type().name()?
			);
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(frozen, module = "aspartik.b3.tree")]
pub struct Leaf(usize);

#[pymethods]
impl Leaf {
	fn __repr__(&self) -> String {
		format!("Leaf({})", self.0)
	}

	fn __eq__(&self, other: Bound<PyAny>) -> Result<bool> {
		if let Ok(node) = other.downcast::<Leaf>() {
			Ok(self.0 == node.get().0)
		} else if other.downcast::<Internal>().is_ok() {
			Ok(false)
		} else {
			py_bail!(
				PyTypeError,
				"Expected `Leaf` or `Internal`, got {}",
				other.get_type().name()?
			);
		}
	}
}

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

		// Sets the weights by walking upwards breadth-first starting
		// with all of the leaves
		const DIFF: f64 = 0.1;
		// The children are all at the position 0
		let mut weights = vec![0.0; num_nodes];
		let mut walk = VecDeque::new();
		for node in parents.iter().take(num_leaves).copied() {
			// the root isn't here because all leaves have a parent
			walk.push_back(node);
		}
		while let Some(node) = walk.pop_front() {
			let idx = 2 * (node - num_leaves);

			let left = weights[children[idx]];
			let right = weights[children[idx + 1]];
			let max = f64::max(left, right);
			let diff = DIFF * (2.0 + rng.random::<f64>());
			weights[node] = max + diff;

			let parent = parents[node];
			if parent != ROOT {
				walk.push_front(parent);
			}
		}

		Self {
			names,

			children: children.into(),
			parents: parents.into(),
			weights: weights.into(),

			updated_edges: Vec::new(),
			updated_nodes: Vec::new(),
		}
	}

	pub(crate) fn accept(&mut self) {
		self.children.accept();
		self.parents.accept();
		self.weights.accept();
		self.clear_updated();
	}

	pub(crate) fn reject(&mut self) {
		self.children.reject();
		self.parents.reject();
		self.weights.reject();
		self.clear_updated();
	}

	fn clear_updated(&mut self) {
		self.updated_edges.clear();
		self.updated_nodes.clear();
	}

	pub(crate) fn edges_to_update(&self) -> Vec<usize> {
		self.updated_edges.clone()
	}

	pub(crate) fn nodes_to_update(&self) -> Vec<Internal> {
		self.walk_nodes(&self.updated_nodes)
	}

	/// A breadth-first order of internals starting from the root.
	pub(crate) fn full_update(&self) -> Vec<Internal> {
		let mut queue = VecDeque::from([self.root()]);
		let mut out = Vec::from([self.root()]);
		while let Some(node) = queue.pop_front() {
			let (left, right) = self.children_of(node);

			if let Some(left) = self.as_internal(left) {
				queue.push_back(left);
				out.push(left);
			}
			if let Some(right) = self.as_internal(right) {
				queue.push_back(right);
				out.push(right);
			}
		}

		out.reverse();
		out
	}

	pub(crate) fn to_lists(
		&self,
		nodes: &[Internal],
	) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
		let mut out_nodes = Vec::new();
		let mut edges = Vec::new();
		let mut children = Vec::new();

		for node in nodes {
			out_nodes.push(node.0);
			let (left, right) = self.children_of(*node);
			children.push(left.0);
			children.push(right.0);

			edges.push(self.edge_index(left));
			edges.push(self.edge_index(right));
		}

		(out_nodes, edges, children)
	}

	fn walk_nodes(&self, nodes: &[Node]) -> Vec<Internal> {
		let mut deq = VecDeque::<Internal>::new();
		let mut set = HashSet::<Internal>::new();

		for node in nodes {
			let mut chain = Vec::new();
			let mut curr =
				self.as_internal(*node).unwrap_or_else(|| {
					self.parent_of(*node).unwrap()
				});

			// Walk up from the starting nodes until the root, stop
			// when we encounter a node we have already walked.
			loop {
				if set.contains(&curr) {
					break;
				}

				set.insert(curr);
				chain.push(curr);

				if let Some(parent) =
					self.parent_of(curr.into())
				{
					curr = parent;
				} else {
					break;
				}
			}

			// Prepend the chain to the deque.  The first chain will
			// insert the root node and walk backwards.  All of the
			// rest will also go in the front, ensuring that
			// children always go befor their parents.
			while let Some(node) = chain.pop() {
				deq.push_front(node);
			}
		}

		deq.into()
	}

	/// Overwrites the child of `edge` with `new_child`.  Only `edge` and
	/// `new_child` changes are recorded, it is presumed that the operator
	/// will call another method for the old child and `new_child`'s parent
	/// edge.
	fn update_edge(&mut self, edge: usize, new_child: Node) {
		let (_, parent) = self.edge_nodes(edge);

		self.children.set(edge, new_child.0);
		self.parents.set(new_child.0, parent.0);

		self.updated_edges.push(edge);

		// `parent` is now the parent of `new_child`, so it'll
		// be updated.  The operator must handle the old node
		// separately.
		self.updated_nodes.push(new_child);
	}

	/// Sets the weight of `node`, recording it and it's parent and child
	/// edges (if it has those).
	pub fn update_weight(&mut self, node: Node, weight: f64) {
		self.weights.set(node.0, weight);
		self.updated_nodes.push(node);

		if self.parent_of(node).is_some() {
			self.updated_edges.push(self.edge_index(node));
		}
		if let Some(node) = self.as_internal(node) {
			let (left, right) = self.children_of(node);
			self.updated_edges.push(self.edge_index(left));
			self.updated_edges.push(self.edge_index(right));
		}
	}

	/// Doesn't overwrite the old root.
	pub fn update_root(&mut self, node: Node) {
		self.parents.set(node.0, ROOT);
	}

	/// Replaces `child` with `replacement` in respect to `child`'s parent.
	pub fn update_replacement(&mut self, child: Node, replacement: Node) {
		let edge = self.edge_index(child);
		self.update_edge(edge, replacement);
	}

	// TODO: invariants (a can't be a parent of b)
	pub fn swap_parents(&mut self, a: Node, b: Node) {
		assert!(self.parent_of(a).is_some(), "a must not be root");
		assert!(self.parent_of(b).is_some(), "b must not be root");

		let edge_a = self.edge_index(a);
		let edge_b = self.edge_index(b);

		self.update_edge(edge_a, b);
		self.update_edge(edge_b, a);
	}

	pub fn verify(&self) -> Result<()> {
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
				self.weight_of(node.into()) > self.weight_of(left),
				"Node {} ({}) is younger than it's left child {} ({})",
				node.0,
				self.weight_of(node.into()),
				left.0,
				self.weight_of(left),
			);
			ensure!(
				self.weight_of(node.into()) > self.weight_of(right),
				"Node {} ({}) is younger than it's right child {} ({})",
				node.0,
				self.weight_of(node.into()),
				left.0,
				self.weight_of(right),
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
			let (left, right) = self.children_of(node);
			children.insert(left);
			children.insert(right);
		}
		ensure!(children.len() == self.num_nodes() - 1);

		Ok(())
	}

	pub fn num_nodes(&self) -> usize {
		self.weights.len()
	}

	pub fn num_internals(&self) -> usize {
		(self.num_nodes() - 1) / 2
	}

	pub fn num_leaves(&self) -> usize {
		self.num_internals() + 1
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

	/// # Panics
	///
	/// Panics if the tree is malformed and has no root.  This can happen
	/// between the calls to `root` and `update_edge`, for example.
	pub fn root(&self) -> Internal {
		// There must always be a rooted element in the tree.
		let i = self.parents.iter().position(|p| *p == ROOT).unwrap();
		Internal(i)
	}

	pub fn weight_of(&self, node: Node) -> f64 {
		self.weights[node.0]
	}

	pub fn children_of(&self, node: Internal) -> (Node, Node) {
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
	pub fn edge_index(&self, child: Node) -> usize {
		let parent = self.parent_of(child).unwrap();

		if self.children_of(parent).0 == child {
			(parent.0 - self.num_leaves()) * 2
		} else {
			(parent.0 - self.num_leaves()) * 2 + 1
		}
	}

	pub fn edge_distance(&self, edge: usize) -> f64 {
		let (child, parent) = self.edge_nodes(edge);

		self.weight_of(parent.into()) - self.weight_of(child)
	}

	fn edge_nodes(&self, edge: usize) -> (Node, Internal) {
		let parent = edge / 2 + self.num_leaves();
		let child = self.children[edge];

		(Node(child), Internal(parent))
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		Some(self.parents[node.0])
			.take_if(|p| *p != ROOT)
			.map(Internal)
	}

	pub fn is_grandparent(&self, node: Internal) -> bool {
		let (left, right) = self.children_of(node);
		self.is_internal(left) && self.is_internal(right)
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
			if let Some(parent) = self.parent_of(node) {
				distance = self.weight_of(parent.into())
					- self.weight_of(node);
			} else {
				distance = 0.0;
			}

			let name = if self.is_leaf(node) {
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
			let (left, right) = self.children_of(parent);

			tree.add_edge(map[&parent.into()], map[&left]);
			tree.add_edge(map[&parent.into()], map[&right]);

			// set root
			if self.parent_of(parent.into()).is_none() {
				tree.set_root(map[&parent.into()]);
			}
		}

		tree.serialize()
	}
}

impl Serialize for Tree {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(&self.to_newick())
	}
}

impl<'de> Deserialize<'de> for Tree {
	fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		// Parse Newick tree from string
		todo!()
	}
}

macro_rules! make_iterator {
	($name: ident, $t: tt) => {
		#[pyclass(module = "aspartik.b3.tree")]
		struct $name {
			current: usize,
			end: usize,
		}

		impl $name {
			fn new(start: usize, end: usize) -> Self {
				Self {
					current: start,
					end,
				}
			}
		}

		#[pymethods]
		impl $name {
			fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
				this
			}

			fn __next__(&mut self) -> Option<$t> {
				if self.current == self.end {
					return None;
				}

				let out = self.current;
				self.current += 1;
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

fn to_node(obj: Bound<PyAny>) -> Result<Node> {
	if let Ok(internal) = obj.extract::<Internal>() {
		Ok(internal.into())
	} else if let Ok(leaf) = obj.extract::<Leaf>() {
		Ok(leaf.into())
	} else if let Ok(node) = obj.extract::<Node>() {
		Ok(node)
	} else {
		Err(PyTypeError::new_err("Wrong type").into())
	}
}

impl PyTree {
	pub fn inner(&self) -> MutexGuard<Tree> {
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

	fn __repr__(&self) -> String {
		"<b3.Tree>".to_string()
	}

	/// Points `edge` to `node`.
	///
	/// This will only change the child, so the parent (internal node from
	/// which `edge` comes out) will now have `node` as a child.
	///
	/// This function doesn't do any validation, it's up to the operator to
	/// preserve the validity of the tree.
	fn update_edge(
		&self,
		edge: usize,
		new_child: Bound<PyAny>,
	) -> Result<()> {
		let new_child = to_node(new_child)?;
		self.inner().update_edge(edge, new_child);
		Ok(())
	}

	/// Sets the weight of `node` to `weight`.
	fn update_weight(&self, node: Bound<PyAny>, weight: f64) -> Result<()> {
		let node = to_node(node)?;
		self.inner().update_weight(node, weight);
		Ok(())
	}

	/// Makes `node` the root of the tree.
	///
	/// The old root must be regrafted with a separate `update_edge` call.
	fn update_root(&self, node: Bound<PyAny>) -> Result<()> {
		let node = to_node(node)?;
		self.inner().update_root(node);
		Ok(())
	}

	/// Swaps the parents of `a` and `b`.
	///
	/// `a` and `b` must not be a child/parent and neither of them can be a
	/// root node.  If `a` and `b` share the same parent, they switch
	/// polarity (left child becomes the right child and visa versa).
	fn swap_parents(&self, a: Bound<PyAny>, b: Bound<PyAny>) -> Result<()> {
		let (a, b) = (to_node(a)?, to_node(b)?);
		self.inner().swap_parents(a, b);
		Ok(())
	}

	/// The total number of nodes in the tree.
	#[getter]
	fn num_nodes(&self) -> usize {
		self.inner().num_nodes()
	}

	/// The number of internal nodes (those with children).
	#[getter]
	fn num_internals(&self) -> usize {
		self.inner().num_internals()
	}

	/// The number of leaves (leaf nodes, those with data).
	#[getter]
	fn num_leaves(&self) -> usize {
		self.inner().num_leaves()
	}

	/// Returns `True` if `node` is internal.
	fn is_internal(&self, node: Bound<PyAny>) -> Result<bool> {
		let node = to_node(node)?;
		Ok(self.inner().is_internal(node))
	}

	/// Returns `True` if `node` is a leaf.
	fn is_leaf(&self, node: Bound<PyAny>) -> Result<bool> {
		let node = to_node(node)?;
		Ok(self.inner().is_leaf(node))
	}

	/// Converts `node` to the type `Internal` if it is internal, or returns
	/// `None` otherwise.
	fn as_internal(&self, node: Node) -> Option<Internal> {
		self.inner().as_internal(node)
	}

	/// Converts `node` to the type `Leaf` if it is a leaf, or returns
	/// `None` otherwise.
	fn as_leaf(&self, node: Node) -> Option<Leaf> {
		self.inner().as_leaf(node)
	}

	/// Returns the root node.
	fn root(&self) -> Internal {
		self.inner().root()
	}

	/// Returns the weight of a node.
	fn weight_of(&self, node: Bound<PyAny>) -> Result<f64> {
		let node = to_node(node)?;
		Ok(self.inner().weight_of(node))
	}

	/// Returns the `(left, right)` children of a node.
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

	/// Returns the index of an edge from `child` to its parent.
	fn edge_index(&self, child: Bound<PyAny>) -> Result<usize> {
		let child = to_node(child)?;
		Ok(self.inner().edge_index(child))
	}

	/// Returns the length of `edge` (distance between the child and the
	/// parent on that edge).
	fn edge_distance(&self, edge: usize) -> f64 {
		self.inner().edge_distance(edge)
	}

	/// Returns the parent of `node`, or `None` if the node is the root of
	/// the tree.
	fn parent_of(&self, node: Bound<PyAny>) -> Result<Option<Internal>> {
		let node = to_node(node)?;

		Ok(self.inner().parent_of(node))
	}

	/// Returns `True` if both children of `node` are also internal.
	fn is_grandparent(&self, node: Internal) -> bool {
		self.inner().is_grandparent(node)
	}

	/// Returns an iterator over all of the nodes.
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

	fn verify(&self) -> Result<()> {
		self.inner().verify()
	}

	fn newick(&self) -> String {
		self.inner().to_newick()
	}
}

pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
	let m = PyModule::new(py, "tree")?;

	m.add_class::<Leaf>()?;
	m.add_class::<Internal>()?;

	let locals = PyDict::new(py);
	locals.set_item("m", &m)?;
	py.run(c"m.Node = m.Leaf | m.Internal", None, Some(&locals))?;

	Ok(m)
}
