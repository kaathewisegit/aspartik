use anyhow::{ensure, Result};
use pyo3::{intern, prelude::*};

use crate::{
	operator::{Proposal, PyProposal},
	tree::PyTree,
};
use rng::PyRng;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct EpochScale {
	tree: Py<PyTree>,
	factor: f64,
	distribution: PyObject,
	rng: Py<PyRng>,
	weight: f64,
}

#[pymethods]
impl EpochScale {
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
		let func = module.getattr(intern!(py, "scale_on_range"))?;

		let (scale, ratio) = func
			.call1((
				low,
				high,
				self.distribution(py),
				self.rng(py),
			))?
			.extract::<(f64, f64)>()?;

		let mut rng = self.rng.get().inner();

		let x = tree.random_internal(&mut rng);
		let y = tree.random_internal(&mut rng);
		let (weight_x, weight_y) =
			(tree.weight_of(x.into()), tree.weight_of(y.into()));
		let lower = f64::min(weight_x, weight_y);
		let upper = f64::max(weight_x, weight_y);

		let move_to = lower + scale * (upper - lower);
		let delta = move_to - upper;

		let mut num_scaled = 0;

		for node in tree.internals() {
			let weight = tree.weight_of(node.into());
			if lower < weight && weight <= upper {
				let new_weight =
					lower + scale * (weight - lower);
				tree.update_weight(node.into(), new_weight);
				num_scaled += 1;
			} else if weight > upper {
				let new_weight = weight + delta;
				tree.update_weight(node.into(), new_weight);
			}
		}

		if num_scaled < 2 {
			return Ok(Proposal::Reject().into());
		}

		Ok(Proposal::Hastings(ratio).into())
	}
}
