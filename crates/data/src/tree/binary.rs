use anyhow::{Result, anyhow, ensure};
use picoarrow::array::{Array, ArrayUtf8, Nullable};
use rustc_hash::{FxBuildHasher, FxHashSet};

use std::cmp::{max, min};

use super::{Internal, Leaf, Node, ROOT_PARENT};

#[derive(Debug)]
pub struct BinaryRootedTree {
	num_leaves: u32,
	root: u32,
	children: Box<[u32]>,
	parents: Box<[u32]>,
	edge_lengths: Box<[f64]>,
	node_names: ArrayUtf8<Nullable>,
	node_metadata: ArrayUtf8<Nullable>,
	edge_metadata: ArrayUtf8<Nullable>,
}

impl BinaryRootedTree {
	pub fn new(
		num_leaves: u32,
		root: u32,
		children: Box<[u32]>,
		edge_lengths: Box<[f64]>,
		mut node_names: ArrayUtf8<Nullable>,
		mut node_metadata: ArrayUtf8<Nullable>,
		mut edge_metadata: ArrayUtf8<Nullable>,
	) -> Result<Self> {
		ensure!(
			num_leaves >= 2,
			"Expected at least two leaves, got {num_leaves}"
		);

		let num_nodes = num_leaves
			.checked_mul(2)
			.and_then(|value| value.checked_sub(1))
			.ok_or_else(|| {
				anyhow!(
					"The number of nodes does not fit in u32"
				)
			})?;
		let num_edges = num_nodes - 1;
		let num_nodes_usize = usize::try_from(num_nodes)?;
		let num_edges_usize = usize::try_from(num_edges)?;
		ensure!(
			(num_leaves..num_nodes).contains(&root),
			"Root node {root} is not an internal node"
		);

		ensure!(
			children.len() == num_edges_usize,
			"Expected {num_edges} child entries, got {}",
			children.len()
		);
		ensure!(
			edge_lengths.len() == num_edges_usize,
			"Expected {num_edges} edge lengths, got {}",
			edge_lengths.len()
		);
		ensure!(
			node_names.len() == num_nodes_usize,
			"Expected {num_nodes} node names, got {}",
			node_names.len()
		);
		ensure!(
			node_metadata.len() == num_nodes_usize,
			"Expected {num_nodes} node metadata entries, got {}",
			node_metadata.len()
		);
		ensure!(
			edge_metadata.len() == num_edges_usize,
			"Expected {num_edges} edge metadata entries, got {}",
			edge_metadata.len()
		);

		let mut parents = vec![ROOT_PARENT; num_nodes_usize];
		let mut seen = vec![false; num_nodes_usize];

		for (offset, pair) in
			children.as_chunks::<2>().0.iter().enumerate()
		{
			let parent = num_leaves + u32::try_from(offset)?;
			for &child in pair {
				ensure!(
					child < num_nodes,
					"Child {child} of node {parent} is out of range"
				);
				ensure!(
					!seen[child as usize],
					"Node {child} appears as a child more than once"
				);

				seen[child as usize] = true;
				parents[child as usize] = parent;
			}
		}

		for node in 0..num_nodes {
			if node == root {
				ensure!(
					!seen[node as usize],
					"The root appears as a child"
				);
			} else {
				ensure!(
					seen[node as usize],
					"Node {node} is not connected to a parent"
				);
			}
		}

		seen.fill(false);
		let mut stack = vec![root];
		while let Some(node) = stack.pop() {
			seen[node as usize] = true;
			if node >= num_leaves {
				let offset = (node - num_leaves) as usize * 2;
				stack.extend_from_slice(
					&children[offset..offset + 2],
				);
			}
		}
		ensure!(
			seen.iter().all(|value| *value),
			"Not all nodes are reachable from the root"
		);

		node_names.shrink_to_fit();
		node_metadata.shrink_to_fit();
		edge_metadata.shrink_to_fit();

		Ok(Self {
			num_leaves,
			root,
			children,
			parents: parents.into_boxed_slice(),
			edge_lengths,
			node_names,
			node_metadata,
			edge_metadata,
		})
	}

	pub fn num_nodes(&self) -> u32 {
		self.num_leaves * 2 - 1
	}

	pub fn num_leaves(&self) -> u32 {
		self.num_leaves
	}

	pub fn num_internals(&self) -> u32 {
		self.num_leaves - 1
	}

	pub fn num_edges(&self) -> u32 {
		self.num_nodes() - 1
	}

	pub fn root(&self) -> Internal {
		Internal(self.root)
	}

	pub fn nodes(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		(0..self.num_nodes()).map(Node)
	}

	pub fn leaves(&self) -> impl DoubleEndedIterator<Item = Leaf> + use<> {
		(0..self.num_leaves()).map(Leaf)
	}

	pub fn internals(
		&self,
	) -> impl DoubleEndedIterator<Item = Internal> + use<> {
		(self.num_leaves()..self.num_nodes()).map(Internal)
	}

	pub fn edges(&self) -> impl DoubleEndedIterator<Item = Node> + use<> {
		let root = self.root;
		(0..root).chain(root + 1..self.num_nodes()).map(Node)
	}

	pub fn is_leaf(&self, node: Node) -> bool {
		node.0 < self.num_leaves()
	}

	pub fn is_internal(&self, node: Node) -> bool {
		(self.num_leaves()..self.num_nodes()).contains(&node.0)
	}

	pub fn as_leaf(&self, node: Node) -> Option<Leaf> {
		self.is_leaf(node).then_some(Leaf(node.0))
	}

	pub fn as_internal(&self, node: Node) -> Option<Internal> {
		self.is_internal(node).then_some(Internal(node.0))
	}

	pub fn children_of(&self, node: Internal) -> (Node, Node) {
		let offset = (node.0 - self.num_leaves()) as usize * 2;
		(Node(self.children[offset]), Node(self.children[offset + 1]))
	}

	pub fn parent_of(&self, node: Node) -> Option<Internal> {
		let parent = self.parents[node.i()];
		(parent != ROOT_PARENT).then_some(Internal(parent))
	}

	pub fn edge_length(&self, child: Node) -> Option<f64> {
		self.edge_index(child).map(|index| self.edge_lengths[index])
	}

	pub fn name(&self, node: Node) -> Option<&str> {
		self.node_names.get(node.i())
	}

	pub fn node_metadata(&self, node: Node) -> Option<&str> {
		self.node_metadata.get(node.i())
	}

	pub fn edge_metadata(&self, child: Node) -> Option<&str> {
		self.edge_index(child)
			.and_then(|index| self.edge_metadata.get(index))
	}

	fn edge_index(&self, child: Node) -> Option<usize> {
		(child.0 < self.num_nodes() && child.0 != self.root).then(
			|| (child.0 - u32::from(child.0 > self.root)) as usize,
		)
	}

	pub fn leaf_by_name(&self, name: &str) -> Option<Leaf> {
		self.leaves().find(|leaf| {
			self.node_names.get(leaf.0 as usize) == Some(name)
		})
	}

	pub fn mrca(&self, first: Node, second: Node) -> Node {
		let mut left = first.0;
		let mut right = second.0;

		while left != right {
			left = if left == ROOT_PARENT {
				second.0
			} else {
				self.parents[left as usize]
			};
			right = if right == ROOT_PARENT {
				first.0
			} else {
				self.parents[right as usize]
			};
		}

		Node(left)
	}

	pub fn ola(&self) -> Vec<i32> {
		let num_nodes = self.num_nodes();
		let num_leaves = self.num_leaves();

		let mut labels = Vec::from_iter(0..num_leaves as i32);
		labels.resize(num_nodes as usize, 0);

		let mut clade_founder = vec![0; num_nodes as usize];

		for node in self.postorder() {
			if let Some(leaf) = self.as_leaf(node) {
				clade_founder[leaf.i()] = leaf.0
			} else if let Some(internal) = self.as_internal(node) {
				let (left, right) = self.children_of(internal);
				clade_founder[internal.i()] = min(
					clade_founder[left.i()],
					clade_founder[right.i()],
				);
			} else {
				unreachable!()
			}
		}

		let mut clade_splitter = vec![0; num_nodes as usize];
		let mut splitter_to_node = vec![0; num_leaves as usize];
		for node in self.preorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};

			let (left, right) = self.children_of(internal);

			let splitter = max(
				clade_founder[left.i()],
				clade_founder[right.i()],
			);
			clade_splitter[node.i()] = splitter;
			labels[node.i()] = -(splitter as i32);
			splitter_to_node[splitter as usize] = node.0;
		}

		let mut ola = Vec::new();
		let mut forward_to = Vec::from_iter(0..num_nodes);

		for label in (1..num_leaves).rev() {
			let splitter_node =
				Internal(splitter_to_node[label as usize]);
			let (left, right) = self.children_of(splitter_node);

			let sibling = if clade_founder[left.i()] == label {
				right
			} else {
				left
			};

			let mut curr = sibling;
			while forward_to[curr.i()] != curr.0 {
				curr = Node(forward_to[curr.i()]);
			}

			ola.push(labels[curr.i()]);
			forward_to[splitter_node.i()] = curr.0;
		}
		ola.reverse();

		ola
	}

	pub fn robinson_foulds(&self, other: &Self) -> u32 {
		let mut counter = 0;
		let mut labels = vec![0; self.num_nodes() as usize];
		let mut clades = FxHashSet::with_capacity_and_hasher(
			self.num_nodes() as usize,
			FxBuildHasher,
		);

		fn process(
			node: Node,
			tree: &BinaryRootedTree,
			counter: &mut u32,
			labels: &mut [u32],
			clades: &mut FxHashSet<(u32, u32)>,
		) -> (u32, u32, u32) {
			let Some(internal) = tree.as_internal(node) else {
				let label = *counter;
				labels[node.i()] = label;
				*counter += 1;
				return (label, label, 1);
			};

			let (left, right) = tree.children_of(internal);
			let (left_min, _left_max, left_size) =
				process(left, tree, counter, labels, clades);
			let (_right_min, right_max, right_size) =
				process(right, tree, counter, labels, clades);
			let size = left_size + right_size;

			if node != tree.root().into() {
				clades.insert((left_min, right_max));
			}

			(left_min, right_max, size)
		}
		process(
			self.root().into(),
			self,
			&mut counter,
			&mut labels,
			&mut clades,
		);

		let mut num_shared_clades = 0;
		fn process_other(
			node: Node,
			tree: &BinaryRootedTree,
			labels: &[u32],
			clades: &FxHashSet<(u32, u32)>,
			num_shared: &mut u32,
		) -> (u32, u32, u32) {
			let Some(internal) = tree.as_internal(node) else {
				let label = labels[node.i()];
				return (label, label, 1);
			};

			let (left, right) = tree.children_of(internal);
			let (left_min, left_max, left_size) = process_other(
				left, tree, labels, clades, num_shared,
			);
			let (right_min, right_max, right_size) = process_other(
				right, tree, labels, clades, num_shared,
			);
			let size = left_size + right_size;

			let min_val = left_min.min(right_min);
			let max_val = left_max.max(right_max);

			if node != tree.root().into()
				&& max_val - min_val + 1 == size
				&& clades.contains(&(min_val, max_val))
			{
				*num_shared += 1;
			}

			(min_val, max_val, size)
		}
		process_other(
			other.root().into(),
			other,
			&labels,
			&clades,
			&mut num_shared_clades,
		);

		let num_nontrivial = self.num_leaves() - 2;
		2 * (num_nontrivial - num_shared_clades)
	}

	pub fn triplet_distance(&self, other: &Self) -> u128 {
		assert_eq!(self.num_leaves(), other.num_leaves());

		let num_leaves = self.num_leaves();
		if num_leaves < 3 {
			return 0;
		}

		let mut subtree_sizes = vec![0; self.num_nodes() as usize];
		for node in self.postorder() {
			if self.is_leaf(node) {
				subtree_sizes[node.i()] = 1;
			} else {
				let (left, right) = self.children_of(
					self.as_internal(node).unwrap(),
				);
				subtree_sizes[node.i()] = subtree_sizes
					[left.i()]
					+ subtree_sizes[right.i()];
			}
		}

		let mut hdt = TripletHdt::new(other);
		let mut steps = vec![TripletStep::Count(self.root().into())];
		let mut leaves = Vec::new();
		let mut shared = 0;

		while let Some(step) = steps.pop() {
			match step {
				TripletStep::Count(node) => {
					let Some(internal) =
						self.as_internal(node)
					else {
						hdt.set_color(
							node.0,
							TripletColor::None,
						);
						continue;
					};

					let (left, right) =
						self.children_of(internal);
					let (small, large) = if subtree_sizes
						[left.i()]
						<= subtree_sizes[right.i()]
					{
						(left, right)
					} else {
						(right, left)
					};

					steps.push(TripletStep::Count(small));
					steps.push(TripletStep::Color(
						small,
						TripletColor::Red,
					));
					steps.push(TripletStep::Count(large));
					steps.push(TripletStep::Color(
						small,
						TripletColor::None,
					));
					steps.push(TripletStep::AddShared);
					steps.push(TripletStep::Color(
						small,
						TripletColor::Blue,
					));
				}
				TripletStep::Color(root, color) => {
					leaves.clear();
					leaves.push(root);
					while let Some(node) = leaves.pop() {
						if self.is_leaf(node) {
							hdt.set_color(
								node.0, color,
							);
						} else {
							let (left, right) = self.children_of(
								self.as_internal(node).unwrap(),
							);
							leaves.extend([
								left, right,
							]);
						}
					}
				}
				TripletStep::AddShared => {
					shared += hdt.shared()
				}
			}
		}

		choose3(num_leaves) - shared
	}

	pub fn preorder(&self) -> impl Iterator<Item = Node> + '_ {
		Preorder {
			tree: self,
			stack: vec![self.root().into()],
		}
	}

	pub fn postorder(&self) -> impl Iterator<Item = Node> + '_ {
		Postorder {
			tree: self,
			stack: vec![(self.root().into(), false)],
		}
	}

	/// Returns `true` if the children have the same names and ids
	pub fn identical_children(&self, other: &BinaryRootedTree) -> bool {
		self.node_names == other.node_names
	}
}

#[derive(Clone, Copy)]
enum TripletColor {
	None,
	Red,
	Blue,
}

enum TripletStep {
	Count(Node),
	Color(Node, TripletColor),
	AddShared,
}

#[derive(Clone, Copy, Default)]
struct TripletCounts {
	red: u32,
	blue: u32,
	same_red: u64,
	same_blue: u64,
	red_below_blue: u64,
	blue_below_red: u64,
	shared: u128,
}

impl TripletCounts {
	fn leaf(color: TripletColor) -> Self {
		match color {
			TripletColor::None => Self::default(),
			TripletColor::Red => Self {
				red: 1,
				..Self::default()
			},
			TripletColor::Blue => Self {
				blue: 1,
				..Self::default()
			},
		}
	}

	fn merge(lower: Self, upper: Self) -> Self {
		Self {
			red: lower.red + upper.red,
			blue: lower.blue + upper.blue,
			same_red: lower.same_red + upper.same_red,
			same_blue: lower.same_blue + upper.same_blue,
			red_below_blue: upper.red_below_blue
				+ lower.red_below_blue + u64::from(
				lower.red,
			) * u64::from(
				upper.blue,
			),
			blue_below_red: upper.blue_below_red
				+ lower.blue_below_red + u64::from(
				lower.blue,
			) * u64::from(
				upper.red,
			),
			shared: lower.shared
				+ upper.shared + u128::from(choose2(lower.red))
				* u128::from(upper.blue) + u128::from(choose2(
				lower.blue,
			)) * u128::from(
				upper.red,
			) + u128::from(upper.same_red)
				* u128::from(lower.blue) + u128::from(
				upper.same_blue,
			) * u128::from(
				lower.red,
			) + u128::from(lower.red)
				* u128::from(upper.red_below_blue)
				+ u128::from(lower.blue)
					* u128::from(upper.blue_below_red),
		}
	}

	fn attach_internal(lower: Self) -> Self {
		Self {
			red: lower.red,
			blue: lower.blue,
			same_red: choose2(lower.red),
			same_blue: choose2(lower.blue),
			shared: lower.shared,
			..Self::default()
		}
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TripletComponentKind {
	Leaf,
	Internal,
	Path {
		lower: usize,
		upper: usize,
		upper_is_internal: bool,
	},
}

struct TripletComponent {
	kind: TripletComponentKind,
	parent: usize,
	down_closed: bool,
	counts: TripletCounts,
}

impl TripletComponent {
	fn leaf(index: usize) -> Self {
		Self {
			kind: TripletComponentKind::Leaf,
			parent: index,
			down_closed: true,
			counts: TripletCounts::leaf(TripletColor::Red),
		}
	}

	fn internal(index: usize) -> Self {
		Self {
			kind: TripletComponentKind::Internal,
			parent: index,
			down_closed: false,
			counts: TripletCounts::default(),
		}
	}
}

#[derive(Clone, Copy)]
struct TripletEdge {
	up: usize,
	down: usize,
}

struct TripletHdt {
	components: Vec<TripletComponent>,
	root: usize,
}

impl TripletHdt {
	fn new(tree: &BinaryRootedTree) -> Self {
		let mut components =
			Vec::with_capacity(tree.num_nodes() as usize * 2 - 1);
		for node in tree.nodes() {
			let index = node.i();
			components.push(if tree.is_leaf(node) {
				TripletComponent::leaf(index)
			} else {
				TripletComponent::internal(index)
			});
		}

		let mut edges = tree
			.edges()
			.map(|child| TripletEdge {
				up: tree.parent_of(child).unwrap().i(),
				down: child.i(),
			})
			.collect::<Vec<_>>();
		let mut next = Vec::with_capacity(edges.len());
		let mut root = tree.root().i();

		while !edges.is_empty() {
			for edge in edges.drain(..) {
				if components[edge.up].parent != edge.up
					|| components[edge.down].parent
						!= edge.down
				{
					next.push(edge);
					continue;
				}

				let up_kind = components[edge.up].kind;
				let down_kind = components[edge.down].kind;
				let upper_is_internal = up_kind
					== TripletComponentKind::Internal;
				let path_merge = matches!(
					up_kind,
					TripletComponentKind::Path { .. }
				) && matches!(
					down_kind,
					TripletComponentKind::Path { .. }
						| TripletComponentKind::Leaf
				);
				let internal_merge = upper_is_internal
					&& components[edge.down].down_closed;

				if !path_merge && !internal_merge {
					next.push(edge);
					continue;
				}

				let index = components.len();
				let counts = if upper_is_internal {
					TripletCounts::attach_internal(
						components[edge.down].counts,
					)
				} else {
					TripletCounts::merge(
						components[edge.down].counts,
						components[edge.up].counts,
					)
				};
				let down_closed = if upper_is_internal {
					false
				} else {
					components[edge.down].down_closed
				};

				components[edge.up].parent = index;
				components[edge.down].parent = index;
				components.push(TripletComponent {
					kind: TripletComponentKind::Path {
						lower: edge.down,
						upper: edge.up,
						upper_is_internal,
					},
					parent: index,
					down_closed,
					counts,
				});
				root = index;
			}

			for edge in &mut next {
				edge.up = components[edge.up].parent;
				edge.down = components[edge.down].parent;
			}
			std::mem::swap(&mut edges, &mut next);
		}

		Self { components, root }
	}

	fn set_color(&mut self, leaf: u32, color: TripletColor) {
		let mut index = leaf as usize;
		self.components[index].counts = TripletCounts::leaf(color);

		while index != self.root {
			index = self.components[index].parent;
			let TripletComponentKind::Path {
				lower,
				upper,
				upper_is_internal,
			} = self.components[index].kind
			else {
				unreachable!()
			};
			self.components[index].counts = if upper_is_internal {
				TripletCounts::attach_internal(
					self.components[lower].counts,
				)
			} else {
				TripletCounts::merge(
					self.components[lower].counts,
					self.components[upper].counts,
				)
			};
		}
	}

	fn shared(&self) -> u128 {
		self.components[self.root].counts.shared
	}
}

fn choose2(value: u32) -> u64 {
	let value = u64::from(value);
	value * value.saturating_sub(1) / 2
}

fn choose3(value: u32) -> u128 {
	let value = u128::from(value);
	value * value.saturating_sub(1) * value.saturating_sub(2) / 6
}

struct Preorder<'a> {
	tree: &'a BinaryRootedTree,
	stack: Vec<Node>,
}

impl Iterator for Preorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Self::Item> {
		let node = self.stack.pop()?;

		if let Some(internal) = self.tree.as_internal(node) {
			let (left, right) = self.tree.children_of(internal);
			self.stack.push(right);
			self.stack.push(left);
		}

		Some(node)
	}
}

struct Postorder<'a> {
	tree: &'a BinaryRootedTree,
	stack: Vec<(Node, bool)>,
}

impl Iterator for Postorder<'_> {
	type Item = Node;

	fn next(&mut self) -> Option<Self::Item> {
		while let Some((node, children_visited)) = self.stack.pop() {
			if children_visited {
				return Some(node);
			}

			self.stack.push((node, true));
			if let Some(internal) = self.tree.as_internal(node) {
				let (left, right) =
					self.tree.children_of(internal);
				self.stack.push((right, false));
				self.stack.push((left, false));
			}
		}

		None
	}
}
