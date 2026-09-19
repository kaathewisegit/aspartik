use anyhow::{Result, anyhow, bail, ensure};
use buffer::Buffer;
use picoarrow::array::{ArrayUtf8, Nullable};
use smallvec::SmallVec;

use std::{collections::VecDeque, mem};

use super::{BinaryTree, Node, ROOT_PARENT};

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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeData {
	pub length: Option<f64>,
	pub attributes: String,
}

impl EdgeData {
	pub const fn new(length: Option<f64>, attributes: String) -> Self {
		EdgeData { length, attributes }
	}

	pub const fn from_distance(length: f64) -> EdgeData {
		EdgeData::new(Some(length), String::new())
	}

	pub const fn without_distance() -> EdgeData {
		EdgeData::new(None, String::new())
	}
}

#[derive(Debug, Clone)]
pub struct TreeBuilder {
	pub(super) children: Vec<SmallVec<[Node; 2]>>,
	pub(super) parents: Vec<Node>,
	pub(super) edges: Vec<EdgeData>,
	pub(super) nodes: Vec<NodeData>,
	pub(super) root: Node,
}

impl Default for TreeBuilder {
	fn default() -> Self {
		Self::new()
	}
}

impl TreeBuilder {
	pub fn new() -> Self {
		Self::with_root(NodeData::unnamed())
	}

	pub fn with_root(data: NodeData) -> Self {
		Self {
			children: vec![SmallVec::new()],
			parents: vec![Node(ROOT_PARENT)],
			edges: vec![EdgeData::default()],
			nodes: vec![data],
			root: Node(0),
		}
	}

	pub fn num_nodes(&self) -> u32 {
		self.nodes.len() as u32
	}

	pub fn root(&self) -> Node {
		self.root
	}

	pub fn nodes(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		(0..self.num_nodes()).map(Node)
	}

	pub fn contains(&self, node: Node) -> bool {
		node.i() < self.nodes.len()
	}

	pub fn is_leaf(&self, node: Node) -> bool {
		self.children_of(node).is_empty()
	}

	pub fn is_binary(&self) -> bool {
		self.children
			.iter()
			.all(|children| matches!(children.len(), 0 | 2))
	}

	pub fn children_of(&self, node: Node) -> &[Node] {
		self.children[node.i()].as_slice()
	}

	pub fn parent_of(&self, node: Node) -> Option<Node> {
		self.parents
			.get(node.i())
			.copied()
			.filter(|parent| parent.index() != ROOT_PARENT)
	}

	pub fn edge(&self, node: Node) -> &EdgeData {
		&self.edges[node.i()]
	}

	pub fn edge_mut(&mut self, node: Node) -> &mut EdgeData {
		&mut self.edges[node.i()]
	}

	pub fn node(&self, node: Node) -> &NodeData {
		&self.nodes[node.i()]
	}

	pub fn node_mut(&mut self, node: Node) -> &mut NodeData {
		&mut self.nodes[node.i()]
	}

	pub fn hybrid_edges(
		&self,
	) -> impl Iterator<Item = (Node, Node)> + use<'_> {
		self.children.iter().enumerate().flat_map(
			move |(i, children)| {
				let parent = Node(i as u32);
				children.iter()
					.copied()
					.filter(move |&child| {
						self.parents[child.i()]
							!= parent
					})
					.map(move |child| (parent, child))
			},
		)
	}

	pub fn add_node(
		&mut self,
		parent: Node,
		data: NodeData,
		edge: EdgeData,
	) -> Result<Node> {
		self.ensure_valid_node(parent)?;
		let index = u32::try_from(self.nodes.len()).map_err(|_| {
			anyhow!("The number of nodes does not fit in u32")
		})?;
		let node = Node(index);
		self.nodes.push(data);
		self.children.push(SmallVec::new());
		self.parents.push(parent);
		self.edges.push(edge);
		self.children[parent.i()].push(node);
		Ok(node)
	}

	pub fn add_edge(
		&mut self,
		parent: Node,
		child: Node,
		edge: EdgeData,
	) -> Result<()> {
		self.ensure_valid_node(parent)?;
		self.ensure_valid_node(child)?;
		ensure!(child != self.root, "The root cannot have a parent");
		ensure!(
			self.parents[child.i()].index() == ROOT_PARENT,
			"Node {} already has a canonical parent",
			child.index()
		);
		ensure!(
			!self.reaches(child, parent),
			"Adding the edge would create a cycle"
		);

		self.parents[child.i()] = parent;
		self.edges[child.i()] = edge;
		self.children[parent.i()].push(child);
		Ok(())
	}

	pub fn remove_edge(
		&mut self,
		parent: Node,
		child: Node,
	) -> Result<EdgeData> {
		self.ensure_valid_node(parent)?;
		self.ensure_valid_node(child)?;
		ensure!(
			self.parents[child.i()] == parent,
			"Node {} is not a canonical child of node {}",
			child.index(),
			parent.index()
		);

		let index = self.children[parent.i()]
			.iter()
			.position(|&node| node == child)
			.ok_or_else(|| {
				anyhow!(
					"The parent-child relation is inconsistent"
				)
			})?;
		self.children[parent.i()].remove(index);
		self.parents[child.i()] = Node(ROOT_PARENT);
		Ok(mem::take(&mut self.edges[child.i()]))
	}

	pub fn replace_parent(
		&mut self,
		child: Node,
		new_parent: Node,
	) -> Result<()> {
		self.ensure_valid_node(new_parent)?;
		self.ensure_valid_node(child)?;
		ensure!(child != self.root, "The root cannot have a parent");
		let Some(old_parent) = self.parent_of(child) else {
			bail!("Node {} has no parent", child.index())
		};
		if old_parent == new_parent {
			return Ok(());
		}
		ensure!(
			!self.reaches(child, new_parent),
			"Changing the parent would create a cycle"
		);

		let Some(index) = self.children[old_parent.i()]
			.iter()
			.position(|&node| node == child)
		else {
			bail!("The parent-child relation is inconsistent")
		};

		self.children[old_parent.i()].remove(index);
		self.children[new_parent.i()].push(child);
		self.parents[child.i()] = new_parent;
		Ok(())
	}

	pub fn replace_edge(
		&mut self,
		child: Node,
		edge: EdgeData,
	) -> Result<EdgeData> {
		self.ensure_valid_node(child)?;
		ensure!(child != self.root, "The root has no incoming edge");
		let current = &mut self.edges[child.i()];
		Ok(mem::replace(current, edge))
	}

	pub fn add_hybrid_edge(
		&mut self,
		parent: Node,
		child: Node,
	) -> Result<()> {
		self.ensure_valid_node(parent)?;
		self.ensure_valid_node(child)?;
		ensure!(parent != child, "A node cannot be its own parent");
		ensure!(
			self.parents[child.i()] != parent,
			"The edge is already the canonical parent relation"
		);
		ensure!(
			!self.children[parent.i()].contains(&child),
			"The hybrid edge already exists"
		);
		ensure!(
			!self.reaches(child, parent),
			"Adding the hybrid edge would create a cycle"
		);

		self.children[parent.i()].push(child);
		Ok(())
	}

	pub fn remove_hybrid_edge(
		&mut self,
		parent: Node,
		child: Node,
	) -> Result<()> {
		self.ensure_valid_node(parent)?;
		self.ensure_valid_node(child)?;
		ensure!(
			self.parents[child.i()] != parent,
			"The edge is the canonical parent relation"
		);
		let index = self.children[parent.i()]
			.iter()
			.position(|&entry| entry == child)
			.ok_or_else(|| {
				anyhow!("The hybrid edge does not exist")
			})?;
		self.children[parent.i()].remove(index);
		Ok(())
	}

	pub fn set_root(&mut self, node: Node) -> Result<()> {
		self.ensure_valid_node(node)?;
		if node == self.root {
			return Ok(());
		}
		ensure!(
			self.hybrid_edges().next().is_none(),
			"Rerooting a tree with hybrid edges is not supported"
		);
		self.validate()?;

		let mut path = Vec::new();
		let mut current = node;
		while current != self.root {
			let parent =
				self.parent_of(current).ok_or_else(|| {
					anyhow!("The new root is disconnected")
				})?;
			let edge = self.edges[current.i()].clone();
			path.push((current, parent, edge));
			current = parent;
		}

		for (child, parent, edge) in path {
			let index = self.children[parent.i()]
				.iter()
				.position(|&entry| entry == child)
				.ok_or_else(|| {
					anyhow!(
						"The parent-child relation is inconsistent"
					)
				})?;
			self.children[parent.i()].remove(index);
			self.children[child.i()].push(parent);
			self.parents[parent.i()] = child;
			self.edges[parent.i()] = edge;
		}

		self.parents[node.i()] = Node(ROOT_PARENT);
		self.edges[node.i()] = EdgeData::default();
		self.root = node;
		self.validate()
	}

	pub fn validate(&self) -> Result<()> {
		let num_nodes = self.nodes.len();
		ensure!(num_nodes > 0, "Expected at least one node");
		ensure!(self.root.i() < num_nodes, "The root is out of range");
		ensure!(
			self.children.len() == num_nodes
				&& self.parents.len() == num_nodes
				&& self.edges.len() == num_nodes,
			"Tree storage lengths are inconsistent"
		);

		let mut seen = vec![false; num_nodes];
		for (parent, children) in self.children.iter().enumerate() {
			let mut unique = SmallVec::<[Node; 2]>::new();
			for &child in children {
				ensure!(
					child.i() < num_nodes,
					"A child node is out of range"
				);
				ensure!(
					child != self.root,
					"The root appears as a child"
				);
				ensure!(
					!unique.contains(&child),
					"Node {} appears more than once under the same parent",
					child.index()
				);
				unique.push(child);
				if self.parents[child.i()]
					== Node(parent as u32)
				{
					ensure!(
						!seen[child.i()],
						"Node {} appears as a canonical child more than once",
						child.index()
					);
					seen[child.i()] = true;
				}
			}
		}

		for node in self.nodes() {
			if node == self.root {
				ensure!(
					self.parents[node.i()].index()
						== ROOT_PARENT,
					"The root has a parent"
				);
			} else {
				ensure!(
					seen[node.i()],
					"Node {} is disconnected",
					node.index()
				);
				ensure!(
					self.parents[node.i()].index()
						!= ROOT_PARENT,
					"Node {} has no canonical parent",
					node.index()
				);
			}
		}

		let mut reachable = vec![false; num_nodes];
		let mut queue = VecDeque::from([self.root]);
		while let Some(node) = queue.pop_front() {
			ensure!(
				!reachable[node.i()],
				"The canonical tree contains a cycle"
			);
			reachable[node.i()] = true;
			queue.extend(self.children[node.i()]
				.iter()
				.copied()
				.filter(|child| {
					self.parents[child.i()] == node
				}));
		}
		ensure!(
			reachable.into_iter().all(|value| value),
			"Not all nodes are reachable from the root"
		);

		let mut indegrees = vec![0_u32; num_nodes];
		for children in &self.children {
			for child in children {
				indegrees[child.i()] += 1;
			}
		}
		let mut queue = indegrees
			.iter()
			.enumerate()
			.filter_map(|(node, &degree)| {
				(degree == 0).then_some(Node(node as u32))
			})
			.collect::<VecDeque<_>>();
		let mut visited = 0;
		while let Some(node) = queue.pop_front() {
			visited += 1;
			for &child in &self.children[node.i()] {
				indegrees[child.i()] -= 1;
				if indegrees[child.i()] == 0 {
					queue.push_back(child);
				}
			}
		}
		ensure!(
			visited == num_nodes,
			"The tree network contains a cycle"
		);
		Ok(())
	}

	pub fn into_binary(self) -> Result<BinaryTree> {
		BinaryTree::try_from(self)
	}

	fn ensure_valid_node(&self, node: Node) -> Result<()> {
		ensure!(self.contains(node), "Node {} is out of range", node.0);
		Ok(())
	}

	fn reaches(&self, start: Node, target: Node) -> bool {
		let mut seen = vec![false; self.nodes.len()];
		let mut stack = vec![start];
		while let Some(node) = stack.pop() {
			if node == target {
				return true;
			}
			if seen[node.i()] {
				continue;
			}
			seen[node.i()] = true;
			stack.extend(self.children[node.i()].iter().copied());
		}
		false
	}
}

impl TryFrom<TreeBuilder> for BinaryTree {
	type Error = anyhow::Error;

	fn try_from(builder: TreeBuilder) -> Result<Self> {
		builder.validate()?;
		ensure!(builder.is_binary(), "The tree is not binary");

		let leaves = builder
			.nodes()
			.filter(|&node| builder.is_leaf(node))
			.collect::<Vec<_>>();
		ensure!(
			leaves.len() >= 2,
			"Expected at least two leaves, got {}",
			leaves.len()
		);
		let internals = builder
			.nodes()
			.filter(|&node| !builder.is_leaf(node))
			.collect::<Vec<_>>();
		let num_leaves = u32::try_from(leaves.len())?;
		let mut order = leaves;
		order.extend(internals);
		let mut mapping = vec![0_u32; builder.nodes.len()];
		for (new, old) in order.iter().copied().enumerate() {
			mapping[old.i()] = u32::try_from(new)?;
		}

		let mut children = Vec::with_capacity(builder.nodes.len() - 1);
		for &old in &order[num_leaves as usize..] {
			children.extend(builder.children[old.i()]
				.iter()
				.map(|child| mapping[child.i()]));
		}
		let root = mapping[builder.root.i()];
		let mut edge_lengths = vec![0.0; builder.nodes.len() - 1];
		let mut edge_attributes = vec![None; builder.nodes.len() - 1];
		for &old in &order {
			let new = mapping[old.i()];
			if new == root {
				continue;
			}
			let index = (new - u32::from(new > root)) as usize;
			let edge = &builder.edges[old.i()];
			edge_lengths[index] = edge.length.ok_or_else(|| {
				anyhow!(
					"Node {} has no edge length",
					old.index()
				)
			})?;
			edge_attributes[index] = nonempty(&edge.attributes);
		}

		let mut names = ArrayUtf8::<Nullable>::new();
		let mut node_attributes = ArrayUtf8::<Nullable>::new();
		for &old in &order {
			names.push(nonempty(&builder.nodes[old.i()].name))?;
			node_attributes.push(nonempty(
				&builder.nodes[old.i()].attributes,
			))?;
		}
		let mut edge_metadata = ArrayUtf8::<Nullable>::new();
		for attributes in edge_attributes {
			edge_metadata.push(attributes)?;
		}

		Self::new(
			num_leaves,
			root,
			Buffer::from_slice(&children),
			Buffer::from_slice(&edge_lengths),
			names,
			node_attributes,
			edge_metadata,
		)
	}
}

fn nonempty(value: &str) -> Option<&str> {
	(!value.is_empty()).then_some(value)
}
