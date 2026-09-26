use anyhow::{Result, anyhow, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::{basic::CompareOp, prelude::*, types::PyType};

use crate::tree::{
	BinaryTree, Internal, Leaf, Node, SvgOptions as TreeSvgOptions,
	TreeLayout, branch_score, branch_score_matrix,
	builder::{EdgeData, NodeData, TreeBuilder},
	robinson_foulds_matrix, triplet_distance_matrix,
};
use rng::PyRng;

fn node_index(value: &Bound<'_, PyAny>) -> Option<u32> {
	if let Ok(node) = value.cast::<Node>() {
		Some(node.get().u32())
	} else if let Ok(leaf) = value.cast::<Leaf>() {
		Some(leaf.get().u32())
	} else if let Ok(internal) = value.cast::<Internal>() {
		Some(internal.get().u32())
	} else {
		None
	}
}

macro_rules! python_node {
	($type:ty, $name:literal) => {
		#[pymethods]
		impl $type {
			#[getter]
			fn index(&self) -> u32 {
				self.u32()
			}

			fn __index__(&self) -> u32 {
				self.u32()
			}

			fn __repr__(&self) -> String {
				format!("{}({})", $name, self.u32())
			}

			fn __hash__(&self) -> isize {
				self.u32() as isize
			}

			fn __richcmp__(
				&self,
				other: &Bound<'_, PyAny>,
				op: CompareOp,
			) -> bool {
				let Some(index) = node_index(other) else {
					return matches!(op, CompareOp::Ne);
				};
				match op {
					CompareOp::Eq => self.u32() == index,
					CompareOp::Ne => self.u32() != index,
					CompareOp::Lt => self.u32() < index,
					CompareOp::Le => self.u32() <= index,
					CompareOp::Gt => self.u32() > index,
					CompareOp::Ge => self.u32() >= index,
				}
			}
		}
	};
}

python_node!(Node, "Node");
python_node!(Leaf, "Leaf");
python_node!(Internal, "Internal");

#[derive(Debug)]
#[pyclass(name = "TreeBuilder", module = "aspartik.data.tree", frozen)]
#[repr(transparent)]
pub struct PyTreeBuilder {
	inner: Mutex<TreeBuilder>,
}

impl PyTreeBuilder {
	pub fn inner(&self) -> MutexGuard<'_, TreeBuilder> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyTreeBuilder {
	#[new]
	fn new() -> Self {
		Self {
			inner: Mutex::new(TreeBuilder::new()),
		}
	}

	#[classmethod]
	fn from_newick(
		_class: &Bound<'_, PyType>,
		newick: &str,
	) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(TreeBuilder::parse_newick(newick)?),
		})
	}

	#[getter]
	fn num_nodes(&self) -> u32 {
		self.inner().num_nodes()
	}

	#[getter]
	fn root(&self) -> Node {
		self.inner().root()
	}

	fn nodes(&self) -> Vec<Node> {
		self.inner().nodes().collect()
	}

	fn is_leaf(&self, node: &Bound<'_, PyAny>) -> Result<bool> {
		let tree = self.inner();
		Ok(tree.is_leaf(checked_node(node, tree.num_nodes())?))
	}

	fn is_binary(&self) -> bool {
		self.inner().is_binary()
	}

	fn children_of(&self, node: &Bound<'_, PyAny>) -> Result<Vec<Node>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(tree.children_of(node).to_vec())
	}

	fn parent_of(&self, node: &Bound<'_, PyAny>) -> Result<Option<Node>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(tree.parent_of(node))
	}

	fn name(&self, node: &Bound<'_, PyAny>) -> Result<Option<String>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(nonempty(&tree.node(node).name))
	}

	fn node_metadata(
		&self,
		node: &Bound<'_, PyAny>,
	) -> Result<Option<String>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(nonempty(&tree.node(node).attributes))
	}

	fn edge_length(&self, child: &Bound<'_, PyAny>) -> Result<Option<f64>> {
		let tree = self.inner();
		let child = checked_node(child, tree.num_nodes())?;
		Ok(tree.edge(child).length)
	}

	fn edge_metadata(
		&self,
		child: &Bound<'_, PyAny>,
	) -> Result<Option<String>> {
		let tree = self.inner();
		let child = checked_node(child, tree.num_nodes())?;
		Ok(nonempty(&tree.edge(child).attributes))
	}

	fn preorder(&self) -> Result<Vec<Node>> {
		let tree = self.inner();
		tree.validate()?;
		let mut output = Vec::with_capacity(tree.num_nodes() as usize);
		let mut stack = vec![tree.root()];
		while let Some(node) = stack.pop() {
			output.push(node);
			stack.extend(tree
				.children_of(node)
				.iter()
				.rev()
				.filter(|&&child| {
					tree.parent_of(child) == Some(node)
				}));
		}
		Ok(output)
	}

	fn postorder(&self) -> Result<Vec<Node>> {
		let tree = self.inner();
		tree.validate()?;
		let mut output = Vec::with_capacity(tree.num_nodes() as usize);
		let mut stack = vec![(tree.root(), false)];
		while let Some((node, visited)) = stack.pop() {
			if visited {
				output.push(node);
				continue;
			}
			stack.push((node, true));
			stack.extend(tree
				.children_of(node)
				.iter()
				.rev()
				.filter(|&&child| {
					tree.parent_of(child) == Some(node)
				})
				.map(|&child| (child, false)));
		}
		Ok(output)
	}

	#[pyo3(signature = (
		parent,
		name = None,
		length = None,
		node_metadata = None,
		edge_metadata = None
	))]
	fn add_node(
		&self,
		parent: &Bound<'_, PyAny>,
		name: Option<String>,
		length: Option<f64>,
		node_metadata: Option<String>,
		edge_metadata: Option<String>,
	) -> Result<Node> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		tree.add_node(
			parent,
			NodeData::new(
				name.unwrap_or_default(),
				node_metadata.unwrap_or_default(),
			),
			EdgeData::new(
				length,
				edge_metadata.unwrap_or_default(),
			),
		)
	}

	#[pyo3(signature = (parent, child, length = None, metadata = None))]
	fn add_edge(
		&self,
		parent: &Bound<'_, PyAny>,
		child: &Bound<'_, PyAny>,
		length: Option<f64>,
		metadata: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		tree.add_edge(
			parent,
			child,
			EdgeData::new(length, metadata.unwrap_or_default()),
		)
	}

	fn remove_edge(
		&self,
		parent: &Bound<'_, PyAny>,
		child: &Bound<'_, PyAny>,
	) -> Result<(Option<f64>, Option<String>)> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		let edge = tree.remove_edge(parent, child)?;
		Ok((edge.length, nonempty(&edge.attributes)))
	}

	fn replace_parent(
		&self,
		child: &Bound<'_, PyAny>,
		new_parent: &Bound<'_, PyAny>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		let new_parent = checked_node(new_parent, tree.num_nodes())?;
		tree.replace_parent(child, new_parent)
	}

	fn set_root(&self, node: &Bound<'_, PyAny>) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.set_root(node)
	}

	fn set_name(
		&self,
		node: &Bound<'_, PyAny>,
		name: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.node_mut(node).name = name.unwrap_or_default();
		Ok(())
	}

	fn set_node_metadata(
		&self,
		node: &Bound<'_, PyAny>,
		metadata: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.node_mut(node).attributes = metadata.unwrap_or_default();
		Ok(())
	}

	fn set_edge_length(
		&self,
		child: &Bound<'_, PyAny>,
		length: Option<f64>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		tree.edge_mut(child).length = length;
		Ok(())
	}

	fn set_edge_metadata(
		&self,
		child: &Bound<'_, PyAny>,
		metadata: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		tree.edge_mut(child).attributes = metadata.unwrap_or_default();
		Ok(())
	}

	fn add_hybrid_edge(
		&self,
		parent: &Bound<'_, PyAny>,
		child: &Bound<'_, PyAny>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		tree.add_hybrid_edge(parent, child)
	}

	fn remove_hybrid_edge(
		&self,
		parent: &Bound<'_, PyAny>,
		child: &Bound<'_, PyAny>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		tree.remove_hybrid_edge(parent, child)
	}

	fn validate(&self) -> Result<()> {
		self.inner().validate()
	}

	fn to_binary(&self) -> Result<PyBinaryTree> {
		Ok(PyBinaryTree {
			inner: self.inner().clone().into_binary()?,
		})
	}

	fn to_newick(&self) -> Result<String> {
		self.inner().to_newick()
	}

	fn __len__(&self) -> usize {
		self.num_nodes() as usize
	}

	fn __str__(&self) -> Result<String> {
		self.to_newick()
	}
}

#[derive(Debug, Clone, Copy)]
#[pyclass(
	name = "SvgOptions",
	module = "aspartik.data.tree",
	frozen,
	skip_from_py_object
)]
pub struct PySvgOptions {
	inner: TreeSvgOptions,
}

#[pymethods]
impl PySvgOptions {
	#[new]
	#[pyo3(signature = (
		x_scale = 100.0,
		y_scale = 30.0,
		margin = 20.0,
		node_radius = 3.0,
		font_size = 12.0,
		show_names = true
	))]
	fn new(
		x_scale: f64,
		y_scale: f64,
		margin: f64,
		node_radius: f64,
		font_size: f64,
		show_names: bool,
	) -> Self {
		Self {
			inner: TreeSvgOptions {
				x_scale,
				y_scale,
				margin,
				node_radius,
				font_size,
				show_names,
			},
		}
	}
}

#[derive(Debug)]
#[pyclass(name = "BinaryTree", module = "aspartik.data.tree", frozen)]
pub struct PyBinaryTree {
	inner: BinaryTree,
}

#[pymethods]
impl PyBinaryTree {
	#[classmethod]
	fn random(
		_class: &Bound<'_, PyType>,
		num_leaves: u32,
		rng: Py<PyRng>,
	) -> Result<Self> {
		Ok(Self {
			inner: BinaryTree::random(
				num_leaves,
				&mut rng.get().inner(),
			)?,
		})
	}

	#[classmethod]
	fn from_newick(
		_class: &Bound<'_, PyType>,
		newick: &str,
	) -> Result<Self> {
		Ok(Self {
			inner: TreeBuilder::parse_newick(newick)?
				.into_binary()?,
		})
	}

	#[getter]
	fn num_nodes(&self) -> u32 {
		self.inner.num_nodes()
	}

	#[getter]
	fn num_leaves(&self) -> u32 {
		self.inner.num_leaves()
	}

	#[getter]
	fn num_internals(&self) -> u32 {
		self.inner.num_internals()
	}

	#[getter]
	fn num_edges(&self) -> u32 {
		self.inner.num_edges()
	}

	#[getter]
	fn root(&self) -> Internal {
		self.inner.root()
	}

	fn nodes(&self) -> Vec<Node> {
		self.inner.nodes().collect()
	}

	fn leaves(&self) -> Vec<Leaf> {
		self.inner.leaves().collect()
	}

	fn internals(&self) -> Vec<Internal> {
		self.inner.internals().collect()
	}

	fn edges(&self) -> Vec<Node> {
		self.inner.edges().collect()
	}

	fn is_leaf(&self, node: &Bound<'_, PyAny>) -> Result<bool> {
		Ok(self.inner.is_leaf(self.node(node)?))
	}

	fn is_internal(&self, node: &Bound<'_, PyAny>) -> Result<bool> {
		Ok(self.inner.is_internal(self.node(node)?))
	}

	fn children_of(&self, node: &Bound<'_, PyAny>) -> Result<(Node, Node)> {
		let node = self.node(node)?;
		let internal =
			self.inner.as_internal(node).ok_or_else(|| {
				anyhow!("Node {} is a leaf", node.u32())
			})?;
		let [left, right] = self.inner.children_of(internal);
		// We return a tuple instead of an array in Python because the
		// former is more compact
		Ok((left, right))
	}

	fn parent_of(
		&self,
		node: &Bound<'_, PyAny>,
	) -> Result<Option<Internal>> {
		Ok(self.inner.parent_of(self.node(node)?))
	}

	fn name(&self, node: &Bound<'_, PyAny>) -> Result<Option<&str>> {
		Ok(self.inner.name(self.node(node)?))
	}

	fn node_metadata(
		&self,
		node: &Bound<'_, PyAny>,
	) -> Result<Option<&str>> {
		Ok(self.inner.node_metadata(self.node(node)?))
	}

	fn edge_length(&self, child: &Bound<'_, PyAny>) -> Result<Option<f64>> {
		Ok(self.inner.edge_length(self.node(child)?))
	}

	fn edge_metadata(
		&self,
		child: &Bound<'_, PyAny>,
	) -> Result<Option<&str>> {
		Ok(self.inner.edge_metadata(self.node(child)?))
	}

	fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		self.inner.leaf_by_name(name)
	}

	fn nhx(
		&self,
		node: &Bound<'_, PyAny>,
		key: &str,
	) -> Result<Option<&str>> {
		Ok(self.inner.nhx(self.node(node)?, key))
	}

	fn preorder(&self) -> Vec<Node> {
		self.inner.preorder().collect()
	}

	fn postorder(&self) -> Vec<Node> {
		self.inner.postorder().collect()
	}

	fn to_newick(&self) -> Result<String> {
		self.inner.to_newick()
	}

	#[pyo3(signature = (kind = "rectangular", separation = 1.0))]
	fn layout(
		&self,
		kind: &str,
		separation: f64,
	) -> Result<Vec<(f64, f64)>> {
		Ok(self.create_layout(kind, separation)?
			.points()
			.iter()
			.map(|point| (point.x, point.y))
			.collect())
	}

	#[pyo3(signature = (
		kind = "rectangular",
		separation = 1.0,
		options = None,
		node_color = "#222222",
		edge_color = "#222222"
	))]
	fn to_svg(
		&self,
		kind: &str,
		separation: f64,
		options: Option<&PySvgOptions>,
		node_color: &str,
		edge_color: &str,
	) -> Result<String> {
		let layout = self.create_layout(kind, separation)?;
		self.inner.to_svg(
			&layout,
			options.map_or_else(TreeSvgOptions::default, |value| {
				value.inner
			}),
			|_| node_color,
			|_| edge_color,
		)
	}

	fn robinson_foulds(&self, other: &PyBinaryTree) -> u32 {
		self.inner.robinson_foulds(&other.inner)
	}

	fn branch_score(&self, other: &PyBinaryTree) -> Result<f64> {
		branch_score(&self.inner, &other.inner)
	}

	fn triplet_distance(&self, other: &PyBinaryTree) -> Result<u128> {
		ensure!(
			self.inner.num_leaves() == other.inner.num_leaves(),
			"Expected both trees to have {} leaves, got {}",
			self.inner.num_leaves(),
			other.inner.num_leaves()
		);
		Ok(self.inner.triplet_distance(&other.inner))
	}

	fn __len__(&self) -> usize {
		self.num_nodes() as usize
	}

	fn __str__(&self) -> Result<String> {
		self.to_newick()
	}
}

#[pyfunction(name = "branch_score_matrix")]
pub fn py_branch_score_matrix(
	py: Python<'_>,
	trees: Vec<Py<PyBinaryTree>>,
) -> Result<Py<PyAny>> {
	let distances = py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		branch_score_matrix(&trees)
	})?;
	let flat: Vec<f64> = distances.into_iter().flatten().collect();
	let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("d",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

impl PyBinaryTree {
	fn node(&self, node: &Bound<'_, PyAny>) -> Result<Node> {
		checked_node(node, self.inner.num_nodes())
	}

	fn create_layout(
		&self,
		kind: &str,
		separation: f64,
	) -> Result<TreeLayout> {
		match kind {
			"rectangular" => {
				self.inner.rectangular_layout(separation)
			}
			"slanted" => self.inner.slanted_layout(separation),
			"tidy" => self.inner.tidy_layout(separation),
			_ => Err(anyhow!("Unknown tree layout '{kind}'")),
		}
	}
}

/// Returns a flat `array("I")` of length `len(trees)**2`, which represents a
/// matrix
#[pyfunction(name = "robinson_foulds_matrix")]
pub fn py_robinson_foulds_matrix(
	py: Python<'_>,
	trees: Vec<Py<PyBinaryTree>>,
) -> Result<Py<PyAny>> {
	let m = py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		robinson_foulds_matrix(&trees[..])
	})?;
	let flat: Vec<u32> = m.into_iter().flatten().collect();
	let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("I",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

#[pyfunction(name = "triplet_distance_matrix")]
pub fn py_triplet_distance_matrix(
	py: Python<'_>,
	trees: Vec<Py<PyBinaryTree>>,
) -> Result<Py<PyAny>> {
	let distances = py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		triplet_distance_matrix(&trees)
	})?;
	let flat = distances
		.into_iter()
		.flatten()
		.map(u64::try_from)
		.collect::<std::result::Result<Vec<_>, _>>()?;
	let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("Q",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

fn checked_node(value: &Bound<'_, PyAny>, num_nodes: u32) -> Result<Node> {
	let index = node_index(value)
		.ok_or_else(|| anyhow!("Expected Node, Leaf, or Internal"))?;
	ensure!(index < num_nodes, "Node {index} is out of range");
	Ok(Node(index))
}

fn nonempty(value: &str) -> Option<String> {
	(!value.is_empty()).then(|| value.to_owned())
}
