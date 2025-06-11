use anyhow::{ensure, Result};
use pyo3::{intern, prelude::*};

use crate::{
	operator::{Proposal, PyProposal},
	tree::PyTree,
};
use rng::PyRng;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct TreeScale {
	tree: Py<PyTree>,
	factor: f64,
	distribution: PyObject,
	rng: Py<PyRng>,
	weight: f64,
}

#[pymethods]
impl TreeScale {
	#[new]
	fn new(
		tree: Py<PyTree>,
		factor: f64,
		distribution: PyObject,
		rng: Py<PyRng>,
		weight: f64,
	) -> Result<Self> {
		ensure!(
			0.0 < factor && factor < 1.0,
			"factor must be between 0 and 1, got {factor}"
		);

		Ok(Self {
			tree,
			factor,
			distribution,
			rng,
			weight,
		})
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	#[getter]
	fn factor(&self) -> f64 {
		self.factor
	}

	#[getter]
	fn distribution(&self, py: Python) -> PyObject {
		self.distribution.clone_ref(py)
	}

	#[getter]
	fn rng(&self, py: Python) -> Py<PyRng> {
		self.rng.clone_ref(py)
	}

	#[getter]
	fn weight(&self) -> f64 {
		self.weight
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<PyObject> {
		let tuple = (
			self.tree(py),
			self.factor,
			self.distribution(py),
			self.rng(py),
			self.weight,
		)
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn propose(&self, py: Python) -> Result<PyProposal> {
		let mut tree = self.tree.get().inner();

		let (low, high) = (self.factor, 1.0 / self.factor);

		let module = PyModule::import(
			py,
			intern!(py, "aspartik.b3.operators._util"),
		)?;
		let func = module.getattr(intern!(py, "sample_range"))?;

		let scale = func
			.call1((
				low,
				high,
				self.distribution.clone_ref(py),
				self.rng.clone_ref(py),
			))?
			.extract::<f64>()?;

		for node in tree.internals() {
			let new_weight = tree.weight_of(node.into()) * scale;
			tree.update_weight(node.into(), new_weight);
		}

		let ratio = scale.ln() * (tree.num_internals() - 2) as f64;

		Ok(Proposal::Hastings(ratio).into())
	}
}
