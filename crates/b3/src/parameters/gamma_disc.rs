use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::{Parameter, PyReal};
use crate::impl_pyparameter_common;
use verbatim::{DeserializeFrom, Serialize};

#[derive(Debug)]
pub struct GammaDisc {
	value: f64,
	backup: f64,
	derived: Box<[Py<PyReal>]>,
	#[expect(dead_code)]
	shape: Py<PyReal>,
}

impl Serialize for GammaDisc {
	fn serialize<W: verbatim::Write>(&self, _writer: &mut W) -> Result<()> {
		todo!()
	}
}

impl DeserializeFrom for &mut GammaDisc {
	fn deserialize_from<'r, R>(self, _reader: &mut R) -> Result<()>
	where
		R: verbatim::Read<'r>,
	{
		todo!()
	}
}

impl Parameter for GammaDisc {
	fn is_changed(&self) -> bool {
		self.derived[0].get().is_changed()
	}

	fn accept(&mut self) {
		self.backup = self.value;
		for real in &self.derived {
			real.get().accept();
		}
	}

	fn reject(&mut self) {
		self.value = self.backup;
		for real in &self.derived {
			real.get().reject();
		}
	}
}

impl GammaDisc {
	pub fn set(&mut self, new_value: f64) {
		self.value = new_value;
		todo!()
	}

	pub fn scale(&mut self, factor: f64) -> usize {
		self.set(self.value * factor);
		todo!()
	}
}

#[pyclass(module = "aspartik.b3.parameters", name = "GammaDisc")]
pub struct PyGammaDisc {
	inner: Mutex<GammaDisc>,
}

impl_pyparameter_common! {PyGammaDisc, GammaDisc;
	#[new]
	pub fn new(
		py: Python,
		value: f64,
		num_reals: usize,
		shape: Py<PyReal>,
	) -> Result<Self> {
		let values = (0..num_reals)
			.map(|_| Py::new(py, PyReal::new(value)))
			.collect::<PyResult<Vec<_>>>()?;
		let values = values.into_boxed_slice();

		let mut inner = GammaDisc {
			value,
			backup: value,
			derived: values,
			shape,
		};

		inner.set(value);

		Ok(Self {
			inner: Mutex::new(inner),
		})
	}

	fn set(&self, value: f64) {
		self.inner().set(value);
	}

	fn scale(&self, factor: f64) -> usize {
		self.inner().scale(factor)
	}
}
