#![expect(unused)]

use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::{PyCpu4Likelihood, PyLikelihood};
use crate::{likelihood, parameters::PyClassVector};

pub struct HeteroLikelihood {
	likelihoods: Vec<PyLikelihood>,
	classes: Py<PyClassVector>,
}

impl HeteroLikelihood {
	fn new(py: Python, likelihoods: Vec<PyLikelihood>) -> Result<Self> {
		let num_classes = likelihoods.len() as u8;
		let num_patterns = likelihoods[0].num_patterns();
		let classes = Py::new(
			py,
			PyClassVector::new(num_classes, num_patterns),
		)?;

		Ok(Self {
			likelihoods,
			classes,
		})
	}

	fn propose(&self) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.propose()?;
		}
		Ok(())
	}

	fn likelihood(&self) -> Result<f64> {
		for likelihood in &self.likelihoods {
			likelihood.likelihood()?;
		}

		let likelihoods: Vec<Vec<f64>> = self
			.likelihoods
			.iter()
			.map(|l| l.pattern_likelihoods())
			.collect::<Result<_>>()?;

		let classes = &*self.classes.get().inner();

		let mut out = 0.0;

		for i in 0..classes.len() {
			let class = classes[i] as usize;
			out += likelihoods[class][i];
		}

		Ok(out)
	}

	fn accept(&self) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.accept()?;
		}
		Ok(())
	}

	fn reject(&self) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.reject()?;
		}
		Ok(())
	}
}

#[pyclass(
	name = "HeteroLikelihood",
	module = "aspartik.b3.likelihoods",
	frozen
)]
pub struct PyHeteroLikelihood {
	inner: Mutex<HeteroLikelihood>,
}

#[pymethods]
impl PyHeteroLikelihood {
	#[new]
	fn new(py: Python, likelihoods: Vec<PyLikelihood>) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(HeteroLikelihood::new(
				py,
				likelihoods,
			)?),
		})
	}

	pub fn propose(&self) -> Result<()> {
		self.inner.lock().propose()
	}

	pub fn likelihood(&self) -> Result<f64> {
		self.inner.lock().likelihood()
	}

	pub fn accept(&self) -> Result<()> {
		self.inner.lock().accept()
	}

	pub fn reject(&self) -> Result<()> {
		self.inner.lock().reject()
	}

	#[getter]
	pub fn class_vector(&self, py: Python) -> Py<PyClassVector> {
		self.inner.lock().classes.clone_ref(py)
	}

	pub fn num_patterns(&self) -> usize {
		self.inner.lock().classes.get().inner().len()
	}
}
