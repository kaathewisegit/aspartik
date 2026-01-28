use anyhow::Result;

mod class_vector;
mod real;
mod tree;

pub use class_vector::{ClassVector, PyClassVector};
pub use real::{PyReal, Real};
pub use tree::{Internal, Leaf, Node, PyTree, Tree};

pub trait Parameter {
	fn is_changed(&self) -> bool;

	fn dump(&self) -> Result<Vec<u8>>;

	fn load(&mut self, bytes: &[u8]) -> Result<()>;

	fn accept(&mut self);

	fn reject(&mut self);
}
