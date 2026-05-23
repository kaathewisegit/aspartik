use anyhow::Result;
use pyo3::prelude::*;

use super::Likelihood;

#[pyclass(module = "aspartik.b3.likelihoods", frozen)]
pub struct CompoundLikelihood {
	likelihoods: Box<[Likelihood]>,
}

#[pymethods]
impl CompoundLikelihood {
	#[new]
	#[pyo3(signature = (*likelihoods))]
	fn new(likelihoods: Vec<Likelihood>) -> Self {
		Self {
			likelihoods: likelihoods.into(),
		}
	}

	pub fn likelihood(&self) -> Result<f64> {
		self.likelihoods.iter().map(|l| l.likelihood()).sum()
	}

	pub fn accept(&self) -> Result<()> {
		self.likelihoods.iter().try_for_each(|l| l.accept())
	}

	pub fn reject(&self) -> Result<()> {
		self.likelihoods.iter().try_for_each(|l| l.reject())
	}
}
