use anyhow::{Result, anyhow, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::tree::{
	BinaryTree, Node, SvgOptions as TreeSvgOptions, TreeLayout,
	branch_score,
	builder::{EdgeData, NodeData, TreeBuilder},
	distance::robinson_foulds_matrix,
};
use rng::PyRng;

#[derive(Debug)]
#[pyclass(name = "Tree", module = "aspartik.data.tree", frozen)]
#[repr(transparent)]
pub struct PyTree {
	inner: Mutex<TreeBuilder>,
}

impl PyTree {
	pub fn inner(&self) -> MutexGuard<'_, TreeBuilder> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyTree {
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
	fn root(&self) -> u32 {
		self.inner().root().index()
	}

	fn nodes(&self) -> Vec<u32> {
		self.inner().nodes().map(Node::index).collect()
	}

	fn is_leaf(&self, node: u32) -> Result<bool> {
		let tree = self.inner();
		Ok(tree.is_leaf(checked_node(node, tree.num_nodes())?))
	}

	fn is_binary(&self) -> bool {
		self.inner().is_binary()
	}

	fn children_of(&self, node: u32) -> Result<Vec<u32>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(tree.children_of(node)
			.iter()
			.map(|child| child.index())
			.collect())
	}

	fn parent_of(&self, node: u32) -> Result<Option<u32>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(tree.parent_of(node).map(Node::index))
	}

	fn name(&self, node: u32) -> Result<Option<String>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(nonempty(&tree.node(node).name))
	}

	fn node_metadata(&self, node: u32) -> Result<Option<String>> {
		let tree = self.inner();
		let node = checked_node(node, tree.num_nodes())?;
		Ok(nonempty(&tree.node(node).attributes))
	}

	fn edge_length(&self, child: u32) -> Result<Option<f64>> {
		let tree = self.inner();
		let child = checked_node(child, tree.num_nodes())?;
		Ok(tree.edge(child).length)
	}

	fn edge_metadata(&self, child: u32) -> Result<Option<String>> {
		let tree = self.inner();
		let child = checked_node(child, tree.num_nodes())?;
		Ok(nonempty(&tree.edge(child).attributes))
	}

	fn preorder(&self) -> Result<Vec<u32>> {
		let tree = self.inner();
		tree.validate()?;
		let mut output = Vec::with_capacity(tree.num_nodes() as usize);
		let mut stack = vec![tree.root()];
		while let Some(node) = stack.pop() {
			output.push(node.index());
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

	fn postorder(&self) -> Result<Vec<u32>> {
		let tree = self.inner();
		tree.validate()?;
		let mut output = Vec::with_capacity(tree.num_nodes() as usize);
		let mut stack = vec![(tree.root(), false)];
		while let Some((node, visited)) = stack.pop() {
			if visited {
				output.push(node.index());
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
		parent: u32,
		name: Option<String>,
		length: Option<f64>,
		node_metadata: Option<String>,
		edge_metadata: Option<String>,
	) -> Result<u32> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		Ok(tree.add_node(
			parent,
			NodeData::new(
				name.unwrap_or_default(),
				node_metadata.unwrap_or_default(),
			),
			EdgeData::new(
				length,
				edge_metadata.unwrap_or_default(),
			),
		)?
		.index())
	}

	#[pyo3(signature = (parent, child, length = None, metadata = None))]
	fn add_edge(
		&self,
		parent: u32,
		child: u32,
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
		parent: u32,
		child: u32,
	) -> Result<(Option<f64>, Option<String>)> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		let edge = tree.remove_edge(parent, child)?;
		Ok((edge.length, nonempty(&edge.attributes)))
	}

	fn replace_parent(&self, child: u32, new_parent: u32) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		let new_parent = checked_node(new_parent, tree.num_nodes())?;
		tree.replace_parent(child, new_parent)
	}

	fn set_root(&self, node: u32) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.set_root(node)
	}

	fn set_name(&self, node: u32, name: Option<String>) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.node_mut(node).name = name.unwrap_or_default();
		Ok(())
	}

	fn set_node_metadata(
		&self,
		node: u32,
		metadata: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let node = checked_node(node, tree.num_nodes())?;
		tree.node_mut(node).attributes = metadata.unwrap_or_default();
		Ok(())
	}

	fn set_edge_length(
		&self,
		child: u32,
		length: Option<f64>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		tree.edge_mut(child).length = length;
		Ok(())
	}

	fn set_edge_metadata(
		&self,
		child: u32,
		metadata: Option<String>,
	) -> Result<()> {
		let mut tree = self.inner.lock();
		let child = checked_node(child, tree.num_nodes())?;
		tree.edge_mut(child).attributes = metadata.unwrap_or_default();
		Ok(())
	}

	fn add_hybrid_edge(&self, parent: u32, child: u32) -> Result<()> {
		let mut tree = self.inner.lock();
		let parent = checked_node(parent, tree.num_nodes())?;
		let child = checked_node(child, tree.num_nodes())?;
		tree.add_hybrid_edge(parent, child)
	}

	fn remove_hybrid_edge(&self, parent: u32, child: u32) -> Result<()> {
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
	fn root(&self) -> u32 {
		self.inner.root().index()
	}

	fn nodes(&self) -> Vec<u32> {
		self.inner.nodes().map(Node::index).collect()
	}

	fn leaves(&self) -> Vec<u32> {
		self.inner.leaves().map(|leaf| leaf.index()).collect()
	}

	fn internals(&self) -> Vec<u32> {
		self.inner
			.internals()
			.map(|internal| internal.index())
			.collect()
	}

	fn edges(&self) -> Vec<u32> {
		self.inner.edges().map(Node::index).collect()
	}

	fn is_leaf(&self, node: u32) -> Result<bool> {
		Ok(self.inner.is_leaf(self.node(node)?))
	}

	fn is_internal(&self, node: u32) -> Result<bool> {
		Ok(self.inner.is_internal(self.node(node)?))
	}

	fn children_of(&self, node: u32) -> Result<(u32, u32)> {
		let node = self.node(node)?;
		let internal =
			self.inner.as_internal(node).ok_or_else(|| {
				anyhow!("Node {} is a leaf", node.index())
			})?;
		let [left, right] = self.inner.children_of(internal);
		// We return a tuple instead of an array in Python because the
		// former is more compact
		Ok((left.index(), right.index()))
	}

	fn parent_of(&self, node: u32) -> Result<Option<u32>> {
		Ok(self.inner
			.parent_of(self.node(node)?)
			.map(|parent| parent.index()))
	}

	fn name(&self, node: u32) -> Result<Option<&str>> {
		Ok(self.inner.name(self.node(node)?))
	}

	fn node_metadata(&self, node: u32) -> Result<Option<&str>> {
		Ok(self.inner.node_metadata(self.node(node)?))
	}

	fn edge_length(&self, child: u32) -> Result<Option<f64>> {
		Ok(self.inner.edge_length(self.node(child)?))
	}

	fn edge_metadata(&self, child: u32) -> Result<Option<&str>> {
		Ok(self.inner.edge_metadata(self.node(child)?))
	}

	fn leaf_by_name(&self, name: &str) -> Option<u32> {
		self.inner.leaf_by_name(name).map(|leaf| leaf.index())
	}

	fn nhx(&self, node: u32, key: &str) -> Result<Option<&str>> {
		Ok(self.inner.nhx(self.node(node)?, key))
	}

	fn preorder(&self) -> Vec<u32> {
		self.inner.preorder().map(Node::index).collect()
	}

	fn postorder(&self) -> Vec<u32> {
		self.inner.postorder().map(Node::index).collect()
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

impl PyBinaryTree {
	fn node(&self, index: u32) -> Result<Node> {
		checked_node(index, self.inner.num_nodes())
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

#[pyfunction(name = "robinson_foulds_matrix")]
pub fn py_robinson_foulds_matrix(
	py: Python<'_>,
	trees: Vec<Py<PyBinaryTree>>,
) -> Result<Vec<Vec<u32>>> {
	py.detach(move || {
		let trees = trees
			.iter()
			.map(|tree| &tree.get().inner)
			.collect::<Vec<_>>();
		robinson_foulds_matrix(&trees[..])
	})
}

fn checked_node(index: u32, num_nodes: u32) -> Result<Node> {
	ensure!(index < num_nodes, "Node {index} is out of range");
	Ok(Node(index))
}

fn nonempty(value: &str) -> Option<String> {
	(!value.is_empty()).then(|| value.to_owned())
}
