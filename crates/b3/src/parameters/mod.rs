use anyhow::Result;
use pyo3::prelude::*;

use std::io::Write;

use verbatim::{DeserializeFrom, Read, Serialize};

mod class_vector;
mod gamma_disc;
mod real;
mod real_vector;
mod tree;

pub use class_vector::{ClassVector, PyClassVector};
pub use real::{PyReal, Real};
pub use real_vector::{PyRealVector, RealVector};
pub use tree::{Internal, Leaf, Node, PyTree, Tree};

pub trait Parameter
where
	Self: Serialize,
	for<'a> &'a mut Self: DeserializeFrom,
{
	fn is_changed(&self) -> bool;

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

			pub fn load(&self, mut bytes: &[u8]) -> Result<()> {
				self.inner().deserialize_from(&mut bytes)
			}

			pub fn dump(&self) -> Result<Vec<u8>> {
				let mut out = Vec::new();
				self.inner().serialize(&mut out)?;
				Ok(out)
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

	pub fn accept(&self) {
		match self {
			Self::ClassVector(p) => p.get().accept(),
			Self::Real(p) => p.get().accept(),
			Self::RealVector(p) => p.get().accept(),
			Self::Tree(p) => p.get().accept(),
		}
	}

	pub fn reject(&self) {
		match self {
			Self::ClassVector(p) => p.get().reject(),
			Self::Real(p) => p.get().reject(),
			Self::RealVector(p) => p.get().reject(),
			Self::Tree(p) => p.get().reject(),
		}
	}

	pub fn serialize<W: Write + ?Sized>(
		&self,
		writer: &mut W,
	) -> Result<()> {
		match self {
			Self::ClassVector(p) => {
				p.get().inner().serialize(writer)
			}
			Self::Real(p) => p.get().inner().serialize(writer),
			Self::RealVector(p) => {
				p.get().inner().serialize(writer)
			}
			Self::Tree(p) => p.get().inner().serialize(writer),
		}
	}

	pub fn deserialize_from<'r, R>(&self, reader: &mut R) -> Result<()>
	where
		R: Read<'r>,
	{
		match self {
			Self::ClassVector(p) => {
				p.get().inner().deserialize_from(reader)
			}
			Self::Real(p) => {
				p.get().inner().deserialize_from(reader)
			}
			Self::RealVector(p) => {
				p.get().inner().deserialize_from(reader)
			}
			Self::Tree(p) => {
				p.get().inner().deserialize_from(reader)
			}
		}
	}
}
