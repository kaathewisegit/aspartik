use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::Parameter;
use skvec::SkVec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassVector {
	num_classes: u8,
	classes: SkVec<u8>,
}

#[expect(clippy::len_without_is_empty)]
impl ClassVector {
	pub fn new(num_classes: u8, len: usize) -> Self {
		Self {
			num_classes,
			classes: SkVec::repeat(0, len),
		}
	}

	pub fn len(&self) -> usize {
		self.classes.len()
	}
}

impl Parameter for ClassVector {
	fn is_changed(&self) -> bool {
		self.classes.is_changed()
	}

	fn dump(&self) -> Result<Vec<u8>> {
		Ok(rmp_serde::to_vec(self)?)
	}

	fn load(&mut self, bytes: &[u8]) -> Result<()> {
		*self = rmp_serde::from_slice(bytes)?;
		Ok(())
	}

	fn accept(&mut self) {
		self.classes.accept()
	}

	fn reject(&mut self) {
		self.classes.reject()
	}
}

#[pyclass(name = "ClassVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyClassVector {
	inner: Mutex<ClassVector>,
}

impl PyClassVector {
	pub fn inner(&self) -> MutexGuard<'_, ClassVector> {
		self.inner.lock()
	}
}
