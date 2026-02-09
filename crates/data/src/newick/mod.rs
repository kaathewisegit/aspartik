use std::collections::BTreeMap;

mod parse;
#[cfg(feature = "python")]
pub(crate) mod python;
mod serialize;

pub use parse::parse;

pub type NodeIdx = u32;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Node {
	name: String,
	attributes: String,
}

impl Node {
	pub const fn new(name: String, attributes: String) -> Node {
		Node { name, attributes }
	}

	pub fn named<S>(name: S) -> Node
	where
		S: AsRef<str>,
	{
		Node::new(name.as_ref().to_owned(), String::new())
	}

	pub const fn unnamed() -> Node {
		Node::new(String::new(), String::new())
	}

	pub fn with_attributes(mut self, attributes: String) -> Node {
		self.attributes = attributes;
		self
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn attributes(&self) -> &str {
		&self.attributes
	}
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Edge {
	distance: Option<f64>,
	attributes: String,
}

impl Edge {
	pub const fn new(distance: Option<f64>, attributes: String) -> Self {
		Edge {
			distance,
			attributes,
		}
	}

	pub const fn from_distance(distance: f64) -> Edge {
		Edge::new(Some(distance), String::new())
	}

	pub fn distance(&self) -> Option<f64> {
		self.distance
	}

	pub fn attributes(&self) -> &str {
		&self.attributes
	}
}

#[derive(Debug, Clone, Default)]
pub struct Tree {
	edges: BTreeMap<(NodeIdx, NodeIdx), Edge>,
	nodes: Vec<Node>,
	root: Option<NodeIdx>,
}

impl Tree {
	pub fn new() -> Tree {
		Tree::default()
	}

	pub fn root(&self) -> Option<&NodeIdx> {
		self.root.as_ref()
	}

	pub fn set_root(&mut self, node: NodeIdx) {
		self.root = Some(node);
	}

	pub fn children_of(
		&self,
		node: NodeIdx,
	) -> impl DoubleEndedIterator<Item = NodeIdx> {
		self.edges
			.range((node, 0)..(node, self.num_nodes() as u32))
			.map(|((_this, child), _)| *child)
	}

	pub fn edges_of(&self, node: NodeIdx) -> impl Iterator<Item = &Edge> {
		self.edges.range((node, 0)..).map(|(_, edge)| edge)
	}

	pub fn parent_of(&self, node: NodeIdx) -> Option<NodeIdx> {
		self.edges.iter().find_map(|(&(parent, child), _)| {
			if child == node { Some(parent) } else { None }
		})
	}

	pub fn get_node(&self, idx: NodeIdx) -> &Node {
		&self.nodes[idx as usize]
	}

	pub fn add_node(&mut self, node: Node) -> NodeIdx {
		self.nodes.push(node);
		(self.nodes.len() - 1) as u32
	}

	pub fn add_edge(&mut self, from: NodeIdx, to: NodeIdx, edge: Edge) {
		self.edges.insert((from, to), edge);
	}

	pub fn edge_to_parent(&self, node: NodeIdx) -> Option<&Edge> {
		self.edges.iter().find_map(|(&(_parent, child), edge)| {
			if child == node { Some(edge) } else { None }
		})
	}

	pub fn num_nodes(&self) -> usize {
		self.nodes.len()
	}

	pub fn leaves(&self) -> impl Iterator<Item = NodeIdx> {
		(0..(self.num_nodes() as u32))
			.filter(|&node| self.is_leaf(node))
	}

	pub fn is_leaf(&self, node: NodeIdx) -> bool {
		self.children_of(node).count() == 0
	}

	pub fn edge(&self, parent: NodeIdx, child: NodeIdx) -> &Edge {
		self.edges.get(&(parent, child)).unwrap()
	}
}
