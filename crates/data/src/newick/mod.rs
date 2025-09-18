use petgraph::{
	Directed,
	stable_graph::{Edges, Neighbors, StableDiGraph},
};

mod parse;
#[cfg(feature = "python")]
pub(crate) mod python;
mod serialize;

pub use petgraph::stable_graph::NodeIndex;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Node {
	name: String,
	attributes: String,
}

impl Node {
	pub fn new(name: String, attributes: String) -> Node {
		Node { name, attributes }
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

	pub fn distance(&self) -> Option<f64> {
		self.distance
	}

	pub fn attributes(&self) -> &str {
		&self.attributes
	}
}

#[derive(Debug, Clone, Default)]
pub struct Tree {
	graph: StableDiGraph<Node, Edge>,
	root: Option<NodeIndex>,
}

impl Tree {
	pub fn new() -> Tree {
		Tree::default()
	}

	pub fn root(&self) -> Option<&NodeIndex> {
		self.root.as_ref()
	}

	pub fn set_root(&mut self, node: NodeIndex) {
		self.root = Some(node);
	}

	pub fn children_of(&self, node: NodeIndex) -> Neighbors<'_, Edge> {
		self.graph.neighbors(node)
	}

	pub fn edges_of(&self, node: NodeIndex) -> Edges<'_, Edge, Directed> {
		self.graph.edges(node)
	}

	pub fn get_node(&self, idx: NodeIndex) -> &Node {
		&self.graph[idx]
	}

	pub fn add_node(&mut self, node: Node) -> NodeIndex {
		self.graph.add_node(node)
	}

	pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: Edge) {
		self.graph.add_edge(from, to, edge);
	}
}
