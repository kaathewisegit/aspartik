use petgraph::stable_graph::StableDiGraph;

mod parse;
#[cfg(feature = "python")]
pub(crate) mod python;
mod serialize;

pub use petgraph::stable_graph::NodeIndex;

#[derive(Debug, Clone, Default)]
pub struct Node {
	name: String,
	#[allow(dead_code)]
	attributes: String,
	distance: Option<f64>,
}

impl Node {
	pub fn new(name: String, attributes: String, distance: f64) -> Node {
		Node {
			name,
			attributes,
			distance: Some(distance),
		}
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn distance(&self) -> Option<f64> {
		self.distance
	}
}

#[derive(Debug, Clone, Default)]
pub struct Tree {
	graph: StableDiGraph<Node, ()>,
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

	pub fn children_of(&self, node: NodeIndex) -> Vec<NodeIndex> {
		self.graph.neighbors(node).collect()
	}

	pub fn get_node(&self, idx: NodeIndex) -> &Node {
		&self.graph[idx]
	}

	pub fn add_node(&mut self, node: Node) -> NodeIndex {
		self.graph.add_node(node)
	}

	pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex) {
		self.graph.add_edge(from, to, ());
	}
}
