use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{exceptions::PyIndexError, prelude::*, types::PyType};

use std::{fmt::Write as _, io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use sk::{Iter, SkBuf};
use util::py_bail;

#[derive(Debug, Clone)]
pub struct IntVector {
	values: SkBuf<i64>,
}

#[allow(clippy::len_without_is_empty)]
impl IntVector {
	pub fn new(values: Vec<i64>) -> Self {
		Self {
			values: values.into(),
		}
	}

	pub fn len(&self) -> usize {
		self.values.len()
	}

	pub fn set(&mut self, index: usize, value: i64) {
		self.values.set(index, value)
	}

	pub fn iter(&self) -> Iter<'_, i64> {
		self.values.iter()
	}
}

impl Index<usize> for IntVector {
	type Output = i64;

	fn index(&self, index: usize) -> &i64 {
		&self.values[index]
	}
}

impl Parameter for IntVector {
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
			self.set(i, 0);
		}
		self.accept();

		for i in 0..self.len() {
			let value = verbatim::read_i64_le(bytes)?;
			self.set(i, value);
		}

		Ok(())
	}

	fn dump(&self, writer: &mut dyn Write) -> Result<()> {
		for i in 0..self.len() {
			verbatim::write_i64_le(writer, self[i])?;
		}
		Ok(())
	}
}

#[pyclass(name = "IntVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyIntVector {
	inner: Mutex<IntVector>,
}

impl_pyparameter_common!(PyIntVector, IntVector, {
	#[new]
	#[pyo3(signature = (*values))]
	fn new(values: Vec<i64>) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(IntVector::new(values)),
		})
	}

	#[classmethod]
	fn repeat(_cls: Py<PyType>, value: i64, length: usize) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(IntVector::new(vec![value; length])),
		})
	}

	fn is_bound(&self, lower: Option<i64>, upper: Option<i64>) -> bool {
		let this = &*self.inner();
		let mut out = true;
		if let Some(lower) = lower {
			out = this.values.iter().all(|&v| v >= lower);
		}
		if let Some(upper) = upper {
			out &= this.values.iter().all(|&v| v < upper);
		}
		out
	}

	fn __len__(&self) -> usize {
		self.inner().len()
	}

	fn __getitem__(&self, index: usize) -> PyResult<i64> {
		let inner = &*self.inner();
		if index >= inner.len() {
			py_bail!(PyIndexError, "Index out of bounds")
		}

		Ok(inner.values[index])
	}

	fn __setitem__(&self, index: usize, value: i64) {
		self.inner().set(index, value);
	}

	fn __repr__(&self) -> Result<String> {
		let mut out = String::from("RealVector(");
		let inner = self.inner();
		for i in 0..inner.len() {
			write!(out, "{}", inner[i])?;
			if i != inner.len() - 1 {
				out.push_str(", ");
			}
		}
		out.push(')');
		Ok(out)
	}
});
