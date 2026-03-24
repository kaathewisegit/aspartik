use anyhow::Result;
use parking_lot::{MappedMutexGuard, MutexGuard};
use pyo3::prelude::*;

use std::io::Write;

mod class_vector;
mod gamma_disc;
mod real;
mod real_vector;
mod tree;

pub use class_vector::{ClassVector, PyClassVector};
pub use real::{PyReal, Real};
pub use real_vector::{PyRealVector, RealVector};
pub use tree::{Internal, Leaf, Node, PyTree, Tree};

pub trait Parameter {
	fn is_changed(&self) -> bool;

	fn dump(&self, dst: &mut dyn Write) -> Result<()>;

	fn load(&mut self, bytes: &[u8]) -> Result<()>;

	fn accept(&mut self);

	fn reject(&mut self);
}

#[macro_export]
macro_rules! impl_pyparameter_common {
	($pytype:ty, $type:ty $(; $($rest:tt)*)?) => {
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

			pub fn load(&self, bytes: &[u8]) -> Result<()> {
				self.inner().load(bytes)
			}

			pub fn accept(&self) {
				self.inner().accept();
			}

			pub fn reject(&self) {
				self.inner().reject();
			}

			$($($rest)*)?
		}
	};
}

#[derive(FromPyObject, IntoPyObject)]
pub enum PyParameter {
	ClassVector(Py<PyClassVector>),
	Real(Py<PyReal>),
	RealVector(Py<PyRealVector>),
	Tree(Py<PyTree>),
}

impl PyParameter {
	pub fn into_py_any(&self, py: Python) -> Py<PyAny> {
		match self {
			Self::ClassVector(p) => p.clone_ref(py).into(),
			Self::Real(p) => p.clone_ref(py).into(),
			Self::RealVector(p) => p.clone_ref(py).into(),
			Self::Tree(p) => p.clone_ref(py).into(),
		}
	}

	pub fn as_ref(&self) -> MappedMutexGuard<'_, dyn Parameter> {
		match self {
			Self::ClassVector(p) => {
				MutexGuard::map(p.get().inner(), |r| {
					r as &mut dyn Parameter
				})
			}
			Self::Real(p) => {
				MutexGuard::map(p.get().inner(), |r| {
					r as &mut dyn Parameter
				})
			}
			Self::RealVector(p) => {
				MutexGuard::map(p.get().inner(), |r| {
					r as &mut dyn Parameter
				})
			}
			Self::Tree(p) => {
				MutexGuard::map(p.get().inner(), |r| {
					r as &mut dyn Parameter
				})
			}
		}
	}
}
