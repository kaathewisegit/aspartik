use anyhow::Result;
use pyo3::prelude::*;

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

pub enum PyParameter {
	Tree(Py<PyTree>),
	Real(Py<PyReal>),
	ClassVector(Py<PyClassVector>),
}

impl<'py> FromPyObject<'_, 'py> for PyParameter {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		if let Ok(tree) = obj.cast::<PyTree>() {
			Ok(Self::Tree(tree.into()))
		} else if let Ok(real) = obj.cast::<PyReal>() {
			Ok(Self::Real(real.into()))
		} else if let Ok(class_vector) = obj.cast::<PyClassVector>() {
			Ok(Self::ClassVector(class_vector.into()))
		} else {
			todo!("descriptive error")
		}
	}
}
