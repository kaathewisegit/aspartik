use pyo3::prelude::*;

use crate::parameters::PyRealVector;
use math::function::gamma::ln_gamma;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct SymmetricDirichlet {
	#[pyo3(get)]
	sum: f64,
	#[pyo3(get)]
	vec: Py<PyRealVector>,
}

#[pymethods]
impl SymmetricDirichlet {
	#[new]
	fn new(vec: Py<PyRealVector>, sum: f64) -> Self {
		Self { sum, vec }
	}

	fn probability(&self) -> f64 {
		let vec = &*self.vec.get().inner();
		let k = vec.len();
		let alpha = self.sum / k as f64;

		// normalization
		let mut out = ln_gamma(self.sum) - k as f64 * ln_gamma(alpha);

		for i in 0..k {
			let x = vec[i];
			out += (alpha - 1.0) * x.ln();
		}

		out
	}

	fn is_changed(&self) -> bool {
		self.vec.get().is_changed()
	}
}
