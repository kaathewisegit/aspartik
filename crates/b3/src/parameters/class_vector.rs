use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::ops::Index;

use super::Parameter;
use crate::impl_pyparameter_common;
use sk::{Iter, SkBuf};
use verbatim::{Deserialize, DeserializeFrom, Serialize};

#[derive(Debug, Clone)]
pub struct ClassVector {
	num_classes: u8,
	classes: SkBuf<u8>,
}

impl Serialize for ClassVector {
	fn serialize<W: verbatim::Write>(&self, writer: &mut W) -> Result<()> {
		for i in 0..self.len() {
			self[i].serialize(writer)?;
		}
		Ok(())
	}
}

impl DeserializeFrom for &mut ClassVector {
	fn deserialize_from<'r, R>(self, reader: &mut R) -> Result<()>
	where
		R: verbatim::Read<'r>,
	{
		for i in 0..self.len() {
			self.set(i, 0);
		}
		self.accept();

		for i in 0..self.len() {
			let c = u8::deserialize(reader)?;
			self.set(i, c);
		}
		Ok(())
	}
}

#[expect(clippy::len_without_is_empty)]
impl ClassVector {
	pub fn new(num_classes: u8, len: usize) -> Self {
		Self {
			num_classes,
			classes: SkBuf::repeat(0, len),
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

impl_pyparameter_common! {PyClassVector, ClassVector;
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
