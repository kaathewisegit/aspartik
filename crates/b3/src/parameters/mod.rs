use anyhow::Result;
use parking_lot::{MappedMutexGuard, MutexGuard};
use pyo3::{ffi::PyObject, prelude::*};

use std::io::Write;

mod class_vector;
mod int_vector;
mod real;
mod real_vector;
mod tree;

pub use class_vector::{ClassVector, PyClassVector};
pub use int_vector::{IntVector, PyIntVector};
pub use real::{PyReal, Real};
pub use real_vector::{PyRealVector, RealVector};
pub use tree::{Internal, Leaf, Node, PyTree, Tree};

pub trait Parameter {
	fn is_changed(&self) -> bool;

	fn accept(&mut self);

	fn reject(&mut self);

	fn load(&mut self, bytes: &mut &[u8]) -> Result<()>;

	fn dump(&self, writer: &mut dyn Write) -> Result<()>;
}

#[macro_export]
macro_rules! impl_pyparameter_common {
	($pytype:ty, $type:ty, { $($rest:item)* }) => {
		impl $pytype {
			pub fn inner(
				&self,
			) -> parking_lot::MutexGuard<'_, $type> {
				self.inner.lock()
			}
		}

		#[pymethods]
		impl $pytype {
			pub fn is_changed(&self) -> bool {
				self.inner().is_changed()
			}

			pub fn load(&self, mut bytes: &[u8]) -> Result<()> {
				self.inner().load(&mut bytes)
			}

			pub fn dump(&self) -> Result<Vec<u8>> {
				let mut out = Vec::new();
				self.inner().dump(&mut out)?;
				Ok(out)
			}

			pub fn accept(&self) {
				self.inner().accept();
			}

			pub fn reject(&self) {
				self.inner().reject();
			}

			$($rest)*
		}
	};
}

#[derive(FromPyObject, IntoPyObject)]
pub enum PyParameter {
	ClassVector(Py<PyClassVector>),
	Real(Py<PyReal>),
	RealVector(Py<PyRealVector>),
	IntVector(Py<PyIntVector>),
	Tree(Py<PyTree>),
}

impl PyParameter {
	pub fn into_py_any(&self, py: Python) -> Py<PyAny> {
		match self {
			Self::ClassVector(p) => p.clone_ref(py).into(),
			Self::Real(p) => p.clone_ref(py).into(),
			Self::RealVector(p) => p.clone_ref(py).into(),
			Self::IntVector(p) => p.clone_ref(py).into(),
			Self::Tree(p) => p.clone_ref(py).into(),
		}
	}

	pub fn as_ptr(&self) -> *mut PyObject {
		match self {
			Self::ClassVector(p) => p.as_ptr(),
			Self::Real(p) => p.as_ptr(),
			Self::RealVector(p) => p.as_ptr(),
			Self::IntVector(p) => p.as_ptr(),
			Self::Tree(p) => p.as_ptr(),
		}
	}

	pub fn as_dyn(&self) -> MappedMutexGuard<'_, dyn Parameter> {
		match self {
			Self::ClassVector(p) => {
				MutexGuard::map(p.get().inner(), |m| {
					m as &mut dyn Parameter
				})
			}
			Self::Real(p) => {
				MutexGuard::map(p.get().inner(), |m| {
					m as &mut dyn Parameter
				})
			}
			Self::RealVector(p) => {
				MutexGuard::map(p.get().inner(), |m| {
					m as &mut dyn Parameter
				})
			}
			Self::IntVector(p) => {
				MutexGuard::map(p.get().inner(), |m| {
					m as &mut dyn Parameter
				})
			}
			Self::Tree(p) => {
				MutexGuard::map(p.get().inner(), |m| {
					m as &mut dyn Parameter
				})
			}
		}
	}
}
