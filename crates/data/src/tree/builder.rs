use anyhow::{Result, anyhow, ensure};
use picoarrow::array::{ArrayUtf8, Nullable};
use smallvec::SmallVec;

use std::collections::VecDeque;

use super::{BinaryTree, Node};

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
	pub(super) children: Vec<SmallVec<[Node; 2]>>,
	pub(super) parents: Vec<Option<Node>>,
	pub(super) edges: Vec<Option<EdgeData>>,
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
			parents: vec![None],
			edges: vec![None],
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
		self.children_of(node).is_some_and(<[Node]>::is_empty)
	}

	pub fn is_binary(&self) -> bool {
		self.children
			.iter()
			.all(|children| matches!(children.len(), 0 | 2))
	}

	pub fn children_of(&self, node: Node) -> Option<&[Node]> {
		self.children.get(node.i()).map(SmallVec::as_slice)
	}

	pub fn parent_of(&self, node: Node) -> Option<Node> {
		self.parents.get(node.i()).copied().flatten()
	}

	pub fn edge(&self, node: Node) -> Option<&EdgeData> {
		self.edges.get(node.i()).and_then(Option::as_ref)
	}

	pub fn edge_mut(&mut self, node: Node) -> Option<&mut EdgeData> {
		self.edges.get_mut(node.i()).and_then(Option::as_mut)
	}

	pub fn node(&self, node: Node) -> Option<&NodeData> {
		self.nodes.get(node.i())
	}

	pub fn node_mut(&mut self, node: Node) -> Option<&mut NodeData> {
		self.nodes.get_mut(node.i())
	}

	pub fn hybrid_edges(
		&self,
	) -> impl Iterator<Item = (Node, Node)> + use<'_> {
		self.children.iter().enumerate().flat_map(
			move |(parent, children)| {
				children.iter().copied().filter_map(
					move |child| {
						(self.parents[child.i()]
							!= Some(Node(
								parent as u32
							)))
						.then_some((
							Node(parent as u32),
							child,
						))
					},
				)
			},
		)
	}

	pub fn add_node(
		&mut self,
		parent: Node,
		data: NodeData,
		edge: EdgeData,
	) -> Result<Node> {
		ensure!(self.contains(parent), "Parent node is out of range");
		let index = u32::try_from(self.nodes.len()).map_err(|_| {
			anyhow!("The number of nodes does not fit in u32")
		})?;
		let node = Node(index);
		self.nodes.push(data);
		self.children.push(SmallVec::new());
		self.parents.push(Some(parent));
		self.edges.push(Some(edge));
		self.children[parent.i()].push(node);
		Ok(node)
	}

	pub fn add_edge(
		&mut self,
		parent: Node,
		child: Node,
		edge: EdgeData,
	) -> Result<()> {
		self.ensure_nodes(parent, child)?;
		ensure!(child != self.root, "The root cannot have a parent");
		ensure!(
			self.parents[child.i()].is_none(),
			"Node {} already has a canonical parent",
			child.index()
		);
		ensure!(
			!self.reaches(child, parent),
			"Adding the edge would create a cycle"
		);

		self.parents[child.i()] = Some(parent);
		self.edges[child.i()] = Some(edge);
		self.children[parent.i()].push(child);
		Ok(())
	}

	pub fn remove_edge(
		&mut self,
		parent: Node,
		child: Node,
	) -> Result<EdgeData> {
		self.ensure_nodes(parent, child)?;
		ensure!(
			self.parents[child.i()] == Some(parent),
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
		self.parents[child.i()] = None;
		self.edges[child.i()].take().ok_or_else(|| {
			anyhow!("The canonical edge has no data")
		})
	}

	pub fn replace_parent(
		&mut self,
		child: Node,
		new_parent: Node,
	) -> Result<()> {
		self.ensure_nodes(child, new_parent)?;
		ensure!(child != self.root, "The root cannot have a parent");
		let old_parent = self.parents[child.i()].ok_or_else(|| {
			anyhow!("Node {} has no parent", child.index())
		})?;
		if old_parent == new_parent {
			return Ok(());
		}
		ensure!(
			!self.reaches(child, new_parent),
			"Changing the parent would create a cycle"
		);

		let index = self.children[old_parent.i()]
			.iter()
			.position(|&node| node == child)
			.ok_or_else(|| {
				anyhow!(
					"The parent-child relation is inconsistent"
				)
			})?;
		self.children[old_parent.i()].remove(index);
		self.children[new_parent.i()].push(child);
		self.parents[child.i()] = Some(new_parent);
		Ok(())
	}

	pub fn spr(&mut self, node: Node, new_parent: Node) -> Result<()> {
		self.replace_parent(node, new_parent)
	}

	pub fn replace_edge(
		&mut self,
		child: Node,
		edge: EdgeData,
	) -> Result<EdgeData> {
		ensure!(self.contains(child), "Child node is out of range");
		ensure!(child != self.root, "The root has no incoming edge");
		let current =
			self.edges[child.i()].as_mut().ok_or_else(|| {
				anyhow!("Node {} has no parent", child.index())
			})?;
		Ok(std::mem::replace(current, edge))
	}

	pub fn add_hybrid_edge(
		&mut self,
		parent: Node,
		child: Node,
	) -> Result<()> {
		self.ensure_nodes(parent, child)?;
		ensure!(parent != child, "A node cannot be its own parent");
		ensure!(
			self.parents[child.i()] != Some(parent),
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
		self.ensure_nodes(parent, child)?;
		ensure!(
			self.parents[child.i()] != Some(parent),
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
		ensure!(self.contains(node), "Root node is out of range");
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
				self.parents[current.i()].ok_or_else(|| {
					anyhow!("The new root is disconnected")
				})?;
			let edge = self.edges[current.i()].take().ok_or_else(
				|| anyhow!("The canonical edge has no data"),
			)?;
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
			self.parents[parent.i()] = Some(child);
			self.edges[parent.i()] = Some(edge);
		}

		self.parents[node.i()] = None;
		self.edges[node.i()] = None;
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
					== Some(Node(parent as u32))
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
					self.parents[node.i()].is_none(),
					"The root has a parent"
				);
				ensure!(
					self.edges[node.i()].is_none(),
					"The root has edge data"
				);
			} else {
				ensure!(
					seen[node.i()],
					"Node {} is disconnected",
					node.index()
				);
				ensure!(
					self.parents[node.i()].is_some(),
					"Node {} has no canonical parent",
					node.index()
				);
				ensure!(
					self.edges[node.i()].is_some(),
					"Node {} has no canonical edge data",
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
					self.parents[child.i()] == Some(node)
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

	fn ensure_nodes(&self, first: Node, second: Node) -> Result<()> {
		ensure!(
			self.contains(first),
			"Node {} is out of range",
			first.index()
		);
		ensure!(
			self.contains(second),
			"Node {} is out of range",
			second.index()
		);
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
			let edge = builder.edges[old.i()].as_ref().ok_or_else(
				|| {
					anyhow!(
						"Node {} has no canonical edge data",
						old.index()
					)
				},
			)?;
			edge_lengths[index] = edge.length;
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
			&children,
			&edge_lengths,
			names,
			node_attributes,
			edge_metadata,
		)
	}
}

fn nonempty(value: &str) -> Option<&str> {
	(!value.is_empty()).then_some(value)
}
