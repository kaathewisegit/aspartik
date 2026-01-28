#![expect(unused)]

use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::PyLikelihood;
use crate::parameters::ClassVector;

pub struct HeteroLikelihood {
	likelihoods: Vec<PyLikelihood>,
	classes: Py<ClassVector>,
}

impl HeteroLikelihood {
	fn propose(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.propose(py)?;
		}
		Ok(())
	}

	fn likelihood(&self, py: Python) -> Result<f64> {
		todo!()
	}

	fn accept(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.accept(py)?;
		}
		Ok(())
	}

	fn reject(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.reject(py)?;
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
