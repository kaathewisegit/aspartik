use pyo3::prelude::*;
use rand::{rngs::OsRng, Rng as _, SeedableRng, TryRngCore};
use rand_pcg::Pcg64;

use std::sync::{Mutex, MutexGuard};

pub type Rng = Pcg64;

#[derive(Debug)]
#[pyclass(name = "RNG", module = "aspartik.rng", frozen)]
#[repr(transparent)]
pub struct PyRng {
	inner: Mutex<Rng>,
}

impl PyRng {
	/// Returns the guard of the Rust rng provider
	///
	/// # Panics
	///
	/// Panics if the mutex holding the inner rng struct has been poisoned.
	pub fn inner(&self) -> MutexGuard<Pcg64> {
		self.inner.lock().unwrap()
	}
}

#[pymethods]
impl PyRng {
	#[new]
	#[pyo3(signature = (seed = None))]
	fn new(seed: Option<u64>) -> PyResult<Self> {
		let seed =
			seed.unwrap_or_else(|| OsRng.try_next_u64().unwrap());

		let inner = Pcg64::seed_from_u64(seed);

		Ok(PyRng {
			inner: Mutex::new(inner),
		})
	}

	#[pyo3(signature = (ratio = 0.5))]
	fn random_bool(&self, ratio: f64) -> bool {
		self.inner().random_bool(ratio)
	}

	fn random_int(&self, lower: i64, upper: i64) -> i64 {
		self.inner().random_range(lower..upper)
	}

	fn random_float(&self) -> f64 {
		self.inner().random()
	}
}

pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_rng_rust_impl")?;

	m.add_class::<PyRng>()?;

	Ok(m)
}
