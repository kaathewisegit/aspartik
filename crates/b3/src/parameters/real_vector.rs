use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{basic::CompareOp, exceptions::PyIndexError, prelude::*};
use serde::{Deserialize, Serialize};

use super::Parameter;
use skvec::SkVec;
use util::py_bail;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealVector {
	values: SkVec<f64>,
}

#[allow(clippy::len_without_is_empty)]
impl RealVector {
	pub fn new(values: Vec<f64>) -> Self {
		Self {
			values: values.into(),
		}
	}

	pub fn len(&self) -> usize {
		self.values.len()
	}
}

impl Parameter for RealVector {
	fn is_changed(&self) -> bool {
		self.values.is_changed()
	}

	fn dump(&self) -> Result<Vec<u8>> {
		Ok(rmp_serde::to_vec(self)?)
	}

	fn load(&mut self, bytes: &[u8]) -> Result<()> {
		*self = rmp_serde::from_slice(bytes)?;
		Ok(())
	}

	fn accept(&mut self) {
		self.values.accept()
	}

	fn reject(&mut self) {
		self.values.reject()
	}
}

#[pyclass(name = "RealVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyRealVector {
	inner: Mutex<RealVector>,
}

impl PyRealVector {
	pub fn inner(&self) -> MutexGuard<'_, RealVector> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyRealVector {
	#[new]
	#[pyo3(signature = (*values))]
	fn new(values: Vec<f64>) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(RealVector::new(values)),
		})
	}

	fn __richcmp__(&self, rhs: f64, op: CompareOp) -> bool {
		let this = &*self.inner();
		match op {
			CompareOp::Lt => this.values.iter().all(|&f| f < rhs),
			CompareOp::Le => this.values.iter().all(|&f| f <= rhs),
			CompareOp::Eq => this.values.iter().all(|&f| f == rhs),
			CompareOp::Ne => this.values.iter().all(|&f| f != rhs),
			CompareOp::Gt => this.values.iter().all(|&f| f > rhs),
			CompareOp::Ge => this.values.iter().all(|&f| f >= rhs),
		}
	}

	fn __len__(&self) -> usize {
		self.inner().len()
	}

	fn __getitem__(&self, index: usize) -> PyResult<f64> {
		let inner = &*self.inner();
		if index >= inner.len() {
			py_bail!(PyIndexError, "Index out of bounds")
		}

		Ok(inner.values[index])
	}

	fn __setitem__(&self, index: usize, value: f64) {
		let inner = &mut *self.inner();
		inner.values.set(index, value);
	}
}
