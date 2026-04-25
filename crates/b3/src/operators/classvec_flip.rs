use pyo3::prelude::*;
use rand::RngExt;

use super::Proposal;
use crate::parameters::PyClassVector;
use rng::PyRng;

#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct ClassvecFlip {
	#[pyo3(get)]
	classvec: Py<PyClassVector>,
	#[pyo3(get)]
	rng: Py<PyRng>,
	#[pyo3(get)]
	weight: f64,
}

#[pymethods]
impl ClassvecFlip {
	#[new]
	fn new(
		classvec: Py<PyClassVector>,
		rng: Py<PyRng>,
		weight: f64,
	) -> Self {
		Self {
			classvec,
			rng,
			weight,
		}
	}

	fn propose(&self) -> Proposal {
		let classvec = &mut *self.classvec.get().inner();
		let rng = &mut *self.rng.get().inner();

		let index = rng.random_range(0..classvec.len());
		let class_at = classvec[index];
		while classvec[index] == class_at {
			let new_class =
				rng.random_range(0..classvec.num_classes());
			classvec.set(index, new_class);
		}

		Proposal::hastings(0.0)
	}
}
