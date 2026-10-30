use anyhow::{Context, Result, anyhow, ensure};
use picoarrow::array::{Array, ArrayUtf8, NonNullable, Nullable};
use smallvec::SmallVec;

use super::{BinaryTree, Node, Tree, newick::parse_newick};
use crate::{TaxonMap, TaxonSet};
use computare_buffer::Buffer;

#[derive(Debug, Clone, Default)]
pub struct TreeBuilder {
	parents: Vec<Node>,
	children: Vec<SmallVec<[Node; 2]>>,
	node_names: Vec<String>,
	node_attributes: Vec<String>,
	edge_lengths: Vec<f64>,
	edge_attributes: Vec<String>,
	root: Option<Node>,
}

impl TreeBuilder {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn add_dummy_node(&mut self) -> Node {
		let idx = self.parents.len() as u32;
		self.parents.push(Node::TOMBSTONE);
		self.children.push(SmallVec::new());
		self.node_names.push(String::new());
		self.node_attributes.push(String::new());
		self.edge_lengths.push(f64::NAN);
		self.edge_attributes.push(String::new());

		Node(idx)
	}

	pub fn add_node(
		&mut self,
		name: &str,
		edge_length: f64,
		parent: Node,
	) -> Node {
		let idx = self.parents.len() as u32;
		let new = Node(idx);

		self.parents.push(parent);
		self.children.push(SmallVec::new());
		self.node_names.push(name.to_owned());
		self.node_attributes.push(String::new());
		self.edge_lengths.push(edge_length);
		self.edge_attributes.push(String::new());

		self.children[parent.usize()].push(new);

		new
	}

	/// Delete all nodes leaving the buffers for reuse
	pub fn clear(&mut self) {
		self.parents.clear();
		self.children.clear();
		self.node_names.clear();
		self.node_attributes.clear();
		self.edge_lengths.clear();
		self.edge_attributes.clear();
		self.root = None;
	}

	pub fn set_root(&mut self, node: Node) {
		self.root = Some(node);
	}

	pub fn num_nodes(&self) -> usize {
		self.parents.len()
	}

	pub fn nodes(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		(0..self.num_nodes() as u32).map(Node)
	}

	// Node setters
	pub fn set_node_parent(&mut self, node: Node, parent: Node) {
		self.parents[node.usize()] = parent;
	}

	pub fn set_node_children(&mut self, node: Node, children: &[Node]) {
		self.children[node.usize()] = SmallVec::from_slice(children);
	}

	pub fn set_node_name(&mut self, node: Node, name: &str) {
		self.node_names[node.usize()].replace_range(.., name);
	}

	pub fn set_node_attribute(&mut self, node: Node, attr: &str) {
		self.node_attributes[node.usize()].replace_range(.., attr);
	}

	pub fn set_edge_length(&mut self, node: Node, length: f64) {
		self.edge_lengths[node.usize()] = length;
	}

	pub fn set_edge_attribute(&mut self, node: Node, attr: &str) {
		self.edge_attributes[node.usize()].replace_range(.., attr);
	}

	// Node getters
	pub fn parent_of(&self, node: Node) -> Option<Node> {
		if self.root.is_some_and(|root| root == node) {
			None
		} else {
			Some(self.parents[node.usize()])
		}
	}

	pub fn children_of(&self, node: Node) -> &[Node] {
		&self.children[node.usize()]
	}

	pub fn name(&self, node: Node) -> &str {
		&self.node_names[node.usize()]
	}

	pub fn name_mut(&mut self, node: Node) -> &mut String {
		&mut self.node_names[node.usize()]
	}

	pub fn node_attribute(&self, node: Node) -> &str {
		&self.node_attributes[node.usize()]
	}

	pub fn edge_length(&self, node: Node) -> Option<f64> {
		let length = self.edge_lengths[node.usize()];
		if length.is_nan() { None } else { Some(length) }
	}

	pub fn edge_attribute(&self, node: Node) -> &str {
		&self.edge_attributes[node.usize()]
	}

	pub fn root(&self) -> Option<Node> {
		self.root
	}

	// checks

	pub fn is_leaf(&self, node: Node) -> bool {
		self.children[node.usize()].is_empty()
	}

	pub fn is_binary(&self) -> bool {
		self.children
			.iter()
			.all(|children| matches!(children.len(), 0 | 2))
	}

	fn make_taxon_set(&self) -> TaxonSet {
		let mut out = ArrayUtf8::<NonNullable>::new();
		for node in self.nodes() {
			if !self.children_of(node).is_empty() {
				continue;
			}
			out.push(self.name_of(node)).unwrap();
		}
		out.into()
	}

	pub fn to_binary_with_taxa(
		&self,
		taxon_set: &TaxonSet,
	) -> Result<BinaryTree> {
		builder_to_binary(self, taxon_set.make_map())
	}

	pub fn to_binary(&self) -> Result<BinaryTree> {
		let taxon_set = self.make_taxon_set();
		self.to_binary_with_taxa(&taxon_set)
	}

	pub fn load_newick(&mut self, newick: &str) -> Result<()> {
		self.clear();
		parse_newick(self, newick)
	}

	pub fn from_newick(newick: &str) -> Result<Self> {
		let mut out = TreeBuilder::new();
		out.load_newick(newick)?;
		Ok(out)
	}
}

pub fn builder_to_binary(
	builder: &TreeBuilder,
	taxon_map: TaxonMap,
) -> Result<BinaryTree> {
	let root = builder.root().ok_or_else(|| anyhow!("tree has no root"))?;
	let num_nodes = builder.num_nodes() as u32;
	let num_leaves = taxon_map.taxon_set().len() as u32;
	ensure!(
		num_leaves >= 2,
		"Expected at least two leaves, got {num_leaves}"
	);
	ensure!(builder.is_binary(), "The tree must be binary");

	let unmapped = Node::TOMBSTONE.0;
	let mut mapping = vec![unmapped; num_nodes as usize];
	let mut next_internal = num_leaves;
	let mut internal_names = ArrayUtf8::<Nullable>::new();

	for node in builder.nodes() {
		if builder.children_of(node).is_empty() {
			let name = builder.name(node);
			let taxon = taxon_map.get(name).with_context(|| {
				anyhow!("Unknown taxon: {name:?}")
			})?;
			mapping[node.usize()] = taxon as u32;
		} else {
			mapping[node.usize()] = next_internal;
			next_internal += 1;
			let name = builder.name(node);
			if name.is_empty() {
				internal_names.push(None)?;
			} else {
				internal_names.push_some(name)?;
			}
		}
	}

	ensure!(
		!mapping.contains(&unmapped),
		"The tree contains unreachable nodes"
	);

	let num_nodes_usize = num_nodes as usize;
	let mut children = Buffer::<u32>::zeroed(num_nodes_usize - 1);
	let mut parents = Buffer::<u32>::repeat(unmapped, num_nodes_usize);
	let mut edge_lengths = Buffer::<f64>::repeat(f64::NAN, num_nodes_usize);

	for node in builder.nodes() {
		let mapped = mapping[node.usize()];
		if let Some(parent) = builder.parent_of(node) {
			parents[mapped as usize] = mapping[parent.usize()];
		}
		if let [left, right] = builder.children_of(node) {
			let offset = (mapped - num_leaves) as usize * 2;
			children[offset] = mapping[left.usize()];
			children[offset + 1] = mapping[right.usize()];
		}
		edge_lengths[mapped as usize] =
			builder.edge_length(node).unwrap_or(0.0);
	}

	let binary = BinaryTree::new(
		num_leaves,
		mapping[root.usize()],
		children,
		parents,
		edge_lengths,
		taxon_map.taxon_set().clone(),
		internal_names,
	);
	Ok(binary)
}

impl Tree for TreeBuilder {
	fn root(&self) -> Option<Node> {
		self.root
	}

	fn parent_of(&self, node: Node) -> Option<Node> {
		self.parent_of(node)
	}

	fn children_of(&self, node: Node) -> &[Node] {
		self.children_of(node)
	}

	fn name_of(&self, node: Node) -> &str {
		self.name(node)
	}

	fn edge_length(&self, node: Node) -> Option<f64> {
		self.edge_length(node)
	}
}
