use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{basic::CompareOp, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Real {
	value: f64,
	backup: f64,
}

impl Real {
	pub fn new(value: f64) -> Self {
		Self {
			value,
			backup: value,
		}
	}

	pub fn value(&self) -> f64 {
		self.value
	}

	pub fn set(&mut self, new_value: f64) {
		self.value = new_value;
	}

	pub fn scale(&mut self, factor: f64) -> usize {
		self.value *= factor;
		1
	}

	pub fn is_changed(&self) -> bool {
		self.value != self.backup
	}

	pub fn accept(&mut self) {
		self.backup = self.value;
	}

	pub fn reject(&mut self) {
		self.value = self.backup;
	}
}

impl PartialEq for Real {
	fn eq(&self, other: &Self) -> bool {
		self.value == other.value
	}
}

impl PartialOrd for Real {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		self.value.partial_cmp(&other.value)
	}
}

#[pyclass(name = "Real", module = "aspartik.b3.parameters", frozen)]
pub struct PyReal {
	inner: Mutex<Real>,
}

impl PyReal {
	pub fn inner(&self) -> MutexGuard<'_, Real> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyReal {
	#[new]
	fn new(value: f64) -> Self {
		Self {
			inner: Mutex::new(Real::new(value)),
		}
	}

	fn value(&self) -> f64 {
		self.inner().value()
	}

	fn set(&self, new_value: f64) {
		self.inner().set(new_value)
	}

	fn is_changed(&self) -> bool {
		self.inner().is_changed()
	}

	fn scale(&self, factor: f64) -> usize {
		self.inner().scale(factor)
	}

	fn accept(&self) {
		self.inner().accept();
	}

	fn reject(&self) {
		self.inner().reject();
	}

	fn dump(&self) -> Result<Vec<u8>> {
		Ok(rmp_serde::to_vec(&*self.inner())?)
	}

	fn load(&self, bytes: &[u8]) -> Result<()> {
		*self.inner() = rmp_serde::from_slice(bytes)?;
		Ok(())
	}

	fn __richcmp__(&self, other: f64, op: CompareOp) -> bool {
		let this = self.value();
		match op {
			CompareOp::Lt => this < other,
			CompareOp::Le => this <= other,
			CompareOp::Eq => this == other,
			CompareOp::Ne => this != other,
			CompareOp::Gt => this > other,
			CompareOp::Ge => this >= other,
		}
	}

	fn __float__(&self) -> f64 {
		self.inner().value()
	}
}
