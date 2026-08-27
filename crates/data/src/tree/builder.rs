use super::Node;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeData {
	pub name: String,
	pub attributes: String,
}

impl NodeData {
	pub const fn new(name: String, attributes: String) -> NodeData {
		NodeData { name, attributes }
	}

	pub fn named(name: impl AsRef<str>) -> NodeData {
		NodeData::new(name.as_ref().to_owned(), String::new())
	}

	pub const fn unnamed() -> NodeData {
		NodeData::new(String::new(), String::new())
	}
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeData {
	pub length: f64,
	pub attributes: String,
}

impl EdgeData {
	pub const fn new(length: f64, attributes: String) -> Self {
		EdgeData { length, attributes }
	}

	pub const fn from_distance(length: f64) -> EdgeData {
		EdgeData::new(length, String::new())
	}
}

#[derive(Debug, Clone)]
pub struct TreeBuilder {
	children: Vec<Vec<Node>>,
	parents: Vec<Node>,
	edges: Vec<EdgeData>,
	nodes: Vec<NodeData>,
	root: Node,
}

impl TreeBuilder {
	pub fn root(&self) -> Node {
		self.root
	}

	pub fn set_root(&mut self, _node: Node) {
		todo!("reverse edges between old root and node");
	}

	pub fn children_of(&self, node: Node) -> &[Node] {
		&self.children[node.i()]
	}

	pub fn parent_of(&self, node: Node) -> Option<Node> {
		if node == self.root {
			None
		} else {
			Some(self.parents[node.i()])
		}
	}

	pub fn edge(&self, node: Node) -> &EdgeData {
		&self.edges[node.i()]
	}

	pub fn node(&self, node: Node) -> &NodeData {
		&self.nodes[node.i()]
	}

	pub fn spr(&mut self, node: Node, new_parent: Node) {
		let old_parent = self.parent_of(node).unwrap();
		let children = &mut self.children[old_parent.i()];
		let idx = children.iter().position(|v| *v == node).unwrap();
		children.swap_remove(idx);

		self.children[new_parent.i()].push(node);

		self.parents[node.i()] = new_parent;
	}
}
