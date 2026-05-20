use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{basic::CompareOp, exceptions::PyIndexError, prelude::*};

use std::{io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use sk::{Iter, SkBuf};
use util::py_bail;
use verbatim::{Deserialize, Serialize};

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
			let value = i64::deserialize(bytes)?;
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

#[pyclass(name = "IntVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyIntVector {
	inner: Mutex<IntVector>,
}

impl_pyparameter_common! {PyIntVector, IntVector;
	#[new]
	#[pyo3(signature = (*values))]
	fn new(values: Vec<i64>) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(IntVector::new(values)),
		})
	}

	fn __richcmp__(&self, rhs: i64, op: CompareOp) -> bool {
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
}
