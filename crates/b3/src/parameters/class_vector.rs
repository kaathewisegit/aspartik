use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use std::{io::Write, ops::Index};

use super::Parameter;
use crate::impl_pyparameter_common;
use skvec::{Iter, SkVec};

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

	pub fn num_classes(&self) -> u8 {
		self.num_classes
	}

	pub fn set(&mut self, index: usize, class: u8) {
		assert!(class < self.num_classes);
		self.classes.set(index, class);
	}

	pub fn iter(&self) -> Iter<'_, u8> {
		self.classes.iter()
	}
}

impl Parameter for ClassVector {
	fn is_changed(&self) -> bool {
		self.classes.is_changed()
	}

	fn dump(&self, dst: &mut dyn Write) -> Result<()> {
		Ok(rmp_serde::encode::write(dst, &self)?)
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

impl Index<usize> for ClassVector {
	type Output = u8;

	fn index(&self, index: usize) -> &u8 {
		&self.classes[index]
	}
}

#[pyclass(name = "ClassVector", module = "aspartik.b3.parameters", frozen)]
pub struct PyClassVector {
	inner: Mutex<ClassVector>,
}

impl_pyparameter_common!(PyClassVector, ClassVector);

#[pymethods]
impl PyClassVector {
	#[new]
	pub fn new(num_classes: u8, len: usize) -> Self {
		Self {
			inner: Mutex::new(ClassVector::new(num_classes, len)),
		}
	}

	pub fn into_list(&self) -> Vec<usize> {
		self.inner().classes.iter().map(|&c| c as usize).collect()
	}
}
