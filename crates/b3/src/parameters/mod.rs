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
	($pytype:ty, $type:ty) => {
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

			pub fn accept(&self) {
				self.inner().accept();
			}

			pub fn reject(&self) {
				self.inner().reject();
			}
		}
	};
}

#[derive(FromPyObject)]
pub enum PyParameter {
	ClassVector(Py<PyClassVector>),
	Real(Py<PyReal>),
	RealVector(Py<PyRealVector>),
	Tree(Py<PyTree>),
}

impl<'py> IntoPyObject<'py> for PyParameter {
	type Target = PyAny;
	type Output = Bound<'py, PyAny>;
	type Error = PyErr;

	fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, PyErr> {
		Ok(match self {
			Self::ClassVector(p) => p.into_bound(py).into_any(),
			Self::Real(p) => p.into_bound(py).into_any(),
			Self::RealVector(p) => p.into_bound(py).into_any(),
			Self::Tree(p) => p.into_bound(py).into_any(),
		})
	}
}

impl PyParameter {
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
