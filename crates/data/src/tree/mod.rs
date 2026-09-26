mod binary;
pub mod builder;
mod distances;
mod newick;
mod parse_newick;
#[cfg(feature = "python")]
pub mod python;
mod render;
mod serialize_newick;

pub use binary::BinaryTree;
pub use distances::{
	branch_score, branch_score_matrix, robinson_foulds_matrix,
	triplet_distance_matrix,
};
pub use parse_newick::parse as parse_newick;
pub use render::{LayoutKind, Point, SvgOptions, TreeLayout};

const ROOT_PARENT: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen, from_py_object)
)]
#[repr(transparent)]
pub struct Node(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen, from_py_object)
)]
#[repr(transparent)]
pub struct Leaf(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
	feature = "python",
	pyo3::pyclass(module = "aspartik.data.tree", frozen, from_py_object)
)]
#[repr(transparent)]
pub struct Internal(u32);

impl Node {
	pub fn u32(self) -> u32 {
		self.0
	}

	pub fn usize(self) -> usize {
		self.0 as usize
	}
}

impl Leaf {
	pub fn u32(self) -> u32 {
		self.0
	}

	pub fn usize(self) -> usize {
		self.0 as usize
	}
}

impl Internal {
	pub fn u32(self) -> u32 {
		self.0
	}

	pub fn usize(self) -> usize {
		self.0 as usize
	}
}

impl From<Leaf> for Node {
	fn from(value: Leaf) -> Self {
		Self(value.0)
	}
}

impl From<Internal> for Node {
	fn from(value: Internal) -> Self {
		Self(value.0)
	}
}
