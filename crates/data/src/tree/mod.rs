mod binary;
pub mod builder;
mod newick;

pub use binary::BinaryTree;

const ROOT_PARENT: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Node(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Leaf(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Internal(u32);

impl Node {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
		self.0 as usize
	}
}

impl Leaf {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
		self.0 as usize
	}
}

impl Internal {
	pub fn index(self) -> u32 {
		self.0
	}

	fn i(self) -> usize {
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
