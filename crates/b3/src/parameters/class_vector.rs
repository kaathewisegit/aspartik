use anyhow::{Result, ensure};
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use sk::{Iter, SkBuf};

#[derive(Debug, Clone)]
pub struct ClassVector {
	num_classes: u32,
	classes: SkBuf<u32>,
}

impl ClassVector {
	pub fn new(num_classes: u32, len: usize) -> Result<Self> {
		ensure!(len > 0, "`ClassVector` must be non-empty");
		ensure!(
			num_classes > 1,
			"`ClassVector` must have at least 2 classes"
		);
		Ok(Self {
			num_classes,
			classes: SkBuf::repeat(0, len),
		})
	}

	#[expect(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.classes.len()
	}

	pub fn num_classes(&self) -> u32 {
		self.num_classes
	}

	pub fn set(&mut self, index: usize, class: u32) {
		assert!(class < self.num_classes);
		self.classes.set(index, class);
	}

	pub fn iter(&self) -> Iter<'_, u32> {
		self.classes.iter()
	}

	pub fn is_changed_at(&self, index: usize) -> bool {
		self.classes.is_changed_at(index)
	}
}

impl Parameter for ClassVector {
	fn is_changed(&self) -> bool {
		self.classes.is_changed()
	}

	fn accept(&mut self) {
		self.classes.accept()
	}

	fn reject(&mut self) {
		self.classes.reject()
	}

	fn load(&mut self, bytes: &mut &[u8]) -> Result<()> {
		for i in 0..self.len() {
			self.set(i, 0);
		}
		self.accept();

		for i in 0..self.len() {
			let c = verbatim::read_u32_le(bytes)?;
			self.set(i, c);
		}
		Ok(())
	}

	fn dump(&self, writer: &mut dyn Write) -> Result<()> {
		for i in 0..self.len() {
			verbatim::write_u32_le(writer, self[i])?;
		}
		Ok(())
	}
}

impl Index<usize> for ClassVector {
	type Output = u32;

	fn index(&self, index: usize) -> &u32 {
		&self.classes[index]
	}
}

#[pyclass(name = "ClassVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyClassVector {
	inner: Mutex<ClassVector>,
}

impl_pyparameter_common!(PyClassVector, ClassVector, {
	#[new]
	pub fn new(num_classes: u32, len: usize) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(ClassVector::new(num_classes, len)?),
		})
	}

	pub fn into_list(&self) -> Vec<usize> {
		self.inner().classes.iter().map(|&c| c as usize).collect()
	}

	pub fn set(&self, index: usize, category: u32) {
		self.inner().set(index, category);
	}

	#[getter]
	pub fn num_classes(&self) -> u32 {
		self.inner().num_classes()
	}

	pub fn __getitem__(&self, index: usize) -> u32 {
		self.inner()[index]
	}

	pub fn __len__(&self) -> usize {
		self.inner().len()
	}
});
