use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{basic::CompareOp, exceptions::PyIndexError, prelude::*};
use serde::{Deserialize, Serialize};

use std::{io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use skvec::{Iter, SkVec};
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

	pub fn set(&mut self, index: usize, value: f64) {
		self.values.set(index, value)
	}

	pub fn iter(&self) -> Iter<'_, f64> {
		self.values.iter()
	}
}

impl Index<usize> for RealVector {
	type Output = f64;

	fn index(&self, index: usize) -> &f64 {
		&self.values[index]
	}
}

impl Parameter for RealVector {
	fn is_changed(&self) -> bool {
		self.values.is_changed()
	}

	fn dump(&self, dst: &mut dyn Write) -> Result<()> {
		Ok(verbatim::to_writer(&self, dst)?)
	}

	fn load(&mut self, bytes: &[u8]) -> Result<()> {
		*self = verbatim::from_slice(bytes)?;
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

impl_pyparameter_common!(PyRealVector, RealVector);

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
		self.inner().set(index, value);
	}
}
