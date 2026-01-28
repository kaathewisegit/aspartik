use pyo3::prelude::*;
use rand::Rng;

use crate::{operator::Proposal, parameters::PyClassVector};
use rng::PyRng;

#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct ClassvecFlip {
	classvec: Py<PyClassVector>,
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

		Proposal::Hastings(0.0)
	}
}
