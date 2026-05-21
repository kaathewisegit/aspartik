use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{exceptions::PyIndexError, prelude::*};

use std::{io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use sk::{Iter, SkBuf};
use util::py_bail;
use verbatim::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct RealVector {
	values: SkBuf<f64>,
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

	fn accept(&mut self) {
		self.values.accept()
	}

	fn reject(&mut self) {
		self.values.reject()
	}

	fn load(&mut self, bytes: &mut &[u8]) -> Result<()> {
		for i in 0..self.len() {
			self.set(i, f64::NAN);
		}
		self.accept();

		for i in 0..self.len() {
			let value = f64::deserialize(bytes)?;
			self.set(i, value);
		}

		Ok(())
	}

	fn dump(&self, writer: &mut dyn Write) -> Result<()> {
		for i in 0..self.len() {
			self[i].serialize(writer)?;
		}
		Ok(())
	}
}

#[pyclass(name = "RealVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyRealVector {
	inner: Mutex<RealVector>,
}

impl_pyparameter_common!(PyRealVector, RealVector, {
	#[new]
	#[pyo3(signature = (*values))]
	fn new(values: Vec<f64>) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(RealVector::new(values)),
		})
	}

	fn is_bound(&self, lower: f64, upper: f64) -> bool {
		let this = &*self.inner();
		this.values.iter().all(|&v| v >= lower)
			&& this.values.iter().all(|&v| v < upper)
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
});
