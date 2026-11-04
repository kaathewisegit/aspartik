mod binary;
pub mod builder;
mod distances;
mod newick;
#[cfg(feature = "python")]
pub mod python;
mod render;

pub use binary::BinaryTree;
pub use builder::TreeBuilder;
pub use distances::{
	branch_score, branch_score_matrix, robinson_foulds_matrix,
	triplet_distance_matrix,
};
pub use render::{LayoutKind, Point, SvgOptions, TreeLayout};

use bytemuck::{Pod, Zeroable};

use std::{fmt, mem, ops::Deref};

pub const TOMBSTONE: u32 = u32::MAX;

#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen)
)]
#[repr(transparent)]
pub struct Node(pub u32);

#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen, from_py_object)
)]
#[repr(transparent)]
pub struct Leaf(pub u32);

#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod,
)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen, from_py_object)
)]
#[repr(transparent)]
pub struct Internal(pub u32);

impl Node {
	const TOMBSTONE: Self = Node(TOMBSTONE);
}

macro_rules! nodes_3 {
	($type:ty) => {
		impl $type {
			pub fn u32(self) -> u32 {
				self.0
			}

			pub fn usize(self) -> usize {
				self.0 as usize
			}
		}
	};
}

nodes_3!(Node);
nodes_3!(Leaf);
nodes_3!(Internal);

macro_rules! nodes_2 {
	($type:ty) => {
		impl Deref for $type {
			type Target = Node;

			fn deref<'a>(&'a self) -> &'a Node {
				// SAFETY: `Leaf`, `Internal`, and `Node` all
				// have the same layout with: `u32` with
				// `repr(transparent)`.  So, casting between
				// them while preserving the lifetimes is safe.
				unsafe {
					mem::transmute::<&'a $type, &'a Node>(
						self,
					)
				}
			}
		}

		impl From<$type> for Node {
			fn from(value: $type) -> Node {
				Node(value.0)
			}
		}
	};
}

nodes_2!(Leaf);
nodes_2!(Internal);

pub trait RootedTree {
	fn num_nodes(&self) -> u32;

	fn root(&self) -> Node;

	fn parent_of(&self, node: Node) -> Option<Node>;

	fn children_of(&self, node: Node) -> &[Node];

	fn name_of(&self, node: Node) -> &str;

	fn edge_length(&self, node: Node) -> Option<f64>;

	fn mrca(&self, mut a: Node, mut b: Node) -> Node {
		let (orig_a, orig_b) = (a, b);
		while a != b {
			a = self.parent_of(a).unwrap_or(orig_b);
			b = self.parent_of(b).unwrap_or(orig_a);
		}
		a
	}

	fn to_newick(&self) -> String {
		struct AsDisplay<'a, T: ?Sized>(&'a T);
		impl<T: RootedTree + ?Sized> fmt::Display for AsDisplay<'_, T> {
			fn fmt(
				&self,
				f: &mut fmt::Formatter<'_>,
			) -> fmt::Result {
				newick::serialize_newick(self.0, f)
			}
		}
		AsDisplay(self).to_string()
	}

	/// Iterator which walks the tree post-order
	fn postorder(&self) -> Postorder<'_, Self>
	where
		Self: Sized,
	{
		Postorder {
			tree: self,
			stack: vec![self.root()],
			prev: None,
		}
	}

	fn preorder(&self) -> Preorder<'_, Self>
	where
		Self: Sized,
	{
		Preorder {
			tree: self,
			stack: vec![self.root()],
		}
	}
}

pub struct Postorder<'a, T> {
	tree: &'a T,
	stack: Vec<Node>,
	prev: Option<Node>,
}

impl<T: RootedTree> Iterator for Postorder<'_, T> {
	type Item = Node;

	fn next(&mut self) -> Option<Node> {
		while let Some(&node) = self.stack.last() {
			let children = self.tree.children_of(node);

			if children.is_empty() {
				self.stack.pop();
				self.prev = Some(node);
				return Some(node);
			}

			if self.prev == children.last().copied() {
				self.stack.pop();
				self.prev = Some(node);
				return Some(node);
			} else {
				self.stack.extend(children.iter().rev());
			}
		}
		None
	}
}

pub struct Preorder<'a, T> {
	tree: &'a T,
	stack: Vec<Node>,
}

impl<T: RootedTree> Iterator for Preorder<'_, T> {
	type Item = Node;

	fn next(&mut self) -> Option<Node> {
		let node = self.stack.pop()?;
		self.stack.extend(self.tree.children_of(node));
		Some(node)
	}
}
