use anyhow::{Result, anyhow, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyType};

use crate::PyTaxonSet;
use crate::tree::{
	BinaryTree, Internal, Leaf, Node, SvgOptions as TreeSvgOptions, Tree,
	TreeBuilder, TreeLayout, branch_score, branch_score_matrix,
	robinson_foulds_matrix, triplet_distance_matrix,
};
use rng::PyRng;
use util::py_bail;

impl<'py> FromPyObject<'_, 'py> for Node {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Node> {
		if let Ok(node) = obj.cast::<Node>() {
			Ok(Node(node.get().0))
		} else if let Ok(internal) = obj.cast::<Internal>() {
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

impl Node {
	pub fn into_pyobject(
		self,
		py: Python,
		num_leaves: u32,
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

macro_rules! python_node {
	($type:ty, $name:literal) => {
		#[pymethods]
		impl $type {
			fn __index__(&self) -> u32 {
				self.u32()
			}

			fn __repr__(&self) -> String {
				format!("{}({})", $name, self.u32())
			}

			fn __hash__(&self) -> isize {
				self.u32() as isize
			}

			fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
				if let Ok(node) = other.cast::<Node>() {
					self.u32() == node.get().u32()
				} else if let Ok(leaf) = other.cast::<Leaf>() {
					self.u32() == leaf.get().u32()
				} else if let Ok(internal) =
					other.cast::<Internal>()
				{
					self.u32() == internal.get().u32()
				} else {
					false
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
			inner: Mutex::new(TreeBuilder::from_newick(newick)?),
		})
	}

	fn load_newick(&self, newick: &str) -> Result<()> {
		let mut inner = self.inner();
		inner.load_newick(newick)
	}

	fn add_dummy_node(&self) -> Node {
		self.inner.lock().add_dummy_node()
	}

	fn set_root(&self, node: Node) {
		self.inner.lock().set_root(node);
	}

	fn set_node_parent(&self, node: Node, parent: Node) {
		self.inner.lock().set_node_parent(node, parent);
	}

	fn set_node_children(&self, node: Node, children: Vec<Node>) {
		self.inner.lock().set_node_children(node, &children);
	}

	#[getter]
	fn num_nodes(&self) -> u32 {
		self.inner().num_nodes() as u32
	}

	#[getter]
	fn root(&self) -> Option<Node> {
		self.inner().root()
	}

	fn nodes(&self) -> Vec<Node> {
		let num_nodes = self.inner().num_nodes() as u32;
		(0..num_nodes).map(Node).collect()
	}

	fn is_binary(&self) -> bool {
		self.inner().is_binary()
	}

	fn children_of(&self, node: Node) -> Vec<Node> {
		self.inner().children_of(node).to_vec()
	}

	fn parent_of(&self, node: Node) -> Option<Node> {
		self.inner().parent_of(node)
	}

	fn name(&self, node: Node) -> Option<String> {
		nonempty(self.inner().name(node))
	}

	fn node_metadata(&self, node: Node) -> Option<String> {
		nonempty(self.inner().node_attribute(node))
	}

	fn edge_length(&self, child: Node) -> Option<f64> {
		self.inner().edge_length(child)
	}

	fn edge_metadata(&self, child: Node) -> Option<String> {
		nonempty(self.inner().edge_attribute(child))
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
		parent: Node,
		name: Option<String>,
		length: Option<f64>,
		node_metadata: Option<String>,
		edge_metadata: Option<String>,
	) -> Node {
		let mut tree = self.inner.lock();
		let node = tree.add_node(
			&name.unwrap_or_default(),
			length.unwrap_or(f64::NAN),
			parent,
		);
		tree.set_node_attribute(
			node,
			&node_metadata.unwrap_or_default(),
		);
		tree.set_edge_attribute(
			node,
			&edge_metadata.unwrap_or_default(),
		);
		node
	}

	fn set_name(&self, node: Node, name: Option<String>) {
		let mut tree = self.inner.lock();
		tree.set_node_name(node, &name.unwrap_or_default());
	}

	fn set_node_metadata(&self, node: Node, metadata: Option<String>) {
		let mut tree = self.inner.lock();
		tree.set_node_attribute(node, &metadata.unwrap_or_default());
	}

	fn set_edge_length(&self, child: Node, length: Option<f64>) {
		let mut tree = self.inner.lock();
		tree.set_edge_length(child, length.unwrap_or(f64::NAN));
	}

	fn set_edge_metadata(&self, child: Node, metadata: Option<String>) {
		self.inner().set_edge_attribute(
			child,
			&metadata.unwrap_or_default(),
		);
	}

	fn is_leaf(&self, node: Node) -> bool {
		self.inner().is_leaf(node)
	}

	#[pyo3(signature = (taxa = None))]
	fn to_binary(&self, taxa: Option<PyTaxonSet>) -> Result<PyBinaryTree> {
		let inner = self.inner();
		let tree = match taxa {
			Some(taxa) => inner.to_binary_with_taxa(&taxa.0)?,
			None => inner.to_binary()?,
		};
		Ok(PyBinaryTree { inner: tree })
	}

	fn to_newick(&self) -> String {
		self.inner().to_newick()
	}

	fn __len__(&self) -> usize {
		self.inner().num_nodes()
	}
}

#[derive(Debug, Clone, Copy)]
#[pyclass(name = "SvgOptions", module = "aspartik.data.tree", frozen)]
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
		taxa: PyTaxonSet,
		rng: Py<PyRng>,
	) -> Result<Self> {
		Ok(Self {
			inner: BinaryTree::random(
				taxa.0.clone(),
				&mut rng.get().inner(),
			)?,
		})
	}

	#[classmethod]
	#[pyo3(signature = (newick, taxa = None))]
	fn from_newick(
		_class: &Bound<'_, PyType>,
		newick: &str,
		taxa: Option<PyTaxonSet>,
	) -> Result<Self> {
		let builder = TreeBuilder::from_newick(newick)?;
		let tree = match taxa {
			Some(taxa) => builder.to_binary_with_taxa(&taxa.0)?,
			None => builder.to_binary()?,
		};
		Ok(Self { inner: tree })
	}

	#[getter]
	fn taxa(&self) -> PyTaxonSet {
		PyTaxonSet(self.inner.taxa().clone())
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

	fn is_leaf(&self, node: Node) -> bool {
		self.inner.is_leaf(node)
	}

	fn is_internal(&self, node: Node) -> bool {
		self.inner.is_internal(node)
	}

	fn children_of(&self, node: Node) -> Result<(Node, Node)> {
		let internal =
			self.inner.as_internal(node).ok_or_else(|| {
				anyhow!("Node {} is a leaf", node.u32())
			})?;
		let [left, right] = self.inner.children_of(internal);
		// We return a tuple instead of an array in Python because the
		// former is more compact
		Ok((left, right))
	}

	fn parent_of(&self, node: Node) -> Option<Internal> {
		self.inner.parent_of(node)
	}

	fn name(&self, node: Node) -> Option<&str> {
		self.inner.name(node)
	}

	fn edge_length(&self, child: Node) -> Option<f64> {
		self.inner.edge_length(child)
	}

	fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		self.inner.leaf_by_name(name)
	}

	fn preorder(&self) -> Vec<Node> {
		self.inner.preorder().collect()
	}

	fn postorder(&self) -> Vec<Node> {
		self.inner.postorder().collect()
	}

	fn to_newick(&self) -> String {
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
	let flat_bytes: &[u8] = bytemuck::cast_slice(distances.as_slice());
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("d",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

impl From<BinaryTree> for PyBinaryTree {
	fn from(tree: BinaryTree) -> Self {
		Self { inner: tree }
	}
}

impl PyBinaryTree {
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
	let distances = py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		robinson_foulds_matrix(&trees[..])
	})?;
	let flat_bytes: &[u8] = bytemuck::cast_slice(distances.as_slice());
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("I",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

#[pyfunction(name = "triplet_distance_matrix")]
#[pyo3(signature = (trees, num_threads = None))]
pub fn py_triplet_distance_matrix(
	py: Python<'_>,
	trees: Vec<Py<PyBinaryTree>>,
	num_threads: Option<usize>,
) -> Result<Py<PyAny>> {
	let distances = py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		triplet_distance_matrix(&trees, num_threads)
	})?;
	let mut flat = Vec::<u64>::with_capacity(distances.num_elements());
	for &distance in distances.as_slice() {
		flat.push(u64::try_from(distance)?);
	}
	let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);
	let array_module = py.import("array")?;
	let py_array = array_module.call_method1("array", ("Q",))?;
	py_array.call_method1("frombytes", (flat_bytes,))?;
	Ok(py_array.unbind())
}

fn nonempty(value: &str) -> Option<String> {
	(!value.is_empty()).then(|| value.to_owned())
}
