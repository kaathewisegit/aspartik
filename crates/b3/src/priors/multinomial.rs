use computare_special::factorial::ln_factorial;
use pyo3::prelude::*;

use crate::parameters::PyClassVector;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct SymmetricMultinomial {
	#[pyo3(get)]
	vec: Py<PyClassVector>,
}

#[pymethods]
impl SymmetricMultinomial {
	#[new]
	fn new(vec: Py<PyClassVector>) -> Self {
		Self { vec }
	}

	fn probability(&self) -> f64 {
		let events = &*self.vec.get().inner();
		let k = events.num_classes();
		let n = events.len();

		let prob_ln = (n as f64).recip().ln() * n as f64;

		let mut counts = vec![0; k as usize]; // histogram
		for i in 0..n {
			if events[i] < k {
				counts[events[i] as usize] += 1;
			}
		}

		let n_factorial = ln_factorial(n as u64);
		let k_factorial_sum: f64 =
			counts.iter().map(|&x| ln_factorial(x)).sum();

		n_factorial - k_factorial_sum + prob_ln
	}

	fn is_changed(&self) -> bool {
		self.vec.get().is_changed()
	}
}
