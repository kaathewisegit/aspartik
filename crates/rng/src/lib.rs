use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::{rngs::OsRng, Rng as _, SeedableRng, TryRngCore};
use rand_pcg::Pcg64;

use util::py_pickle_state_impl;

pub type Rng = Pcg64;

#[derive(Debug)]
#[pyclass(name = "RNG", module = "aspartik.rng", frozen)]
#[repr(transparent)]
pub struct PyRng {
	inner: Mutex<Rng>,
}

impl PyRng {
	pub fn inner(&self) -> MutexGuard<Pcg64> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyRng {
	#[new]
	#[pyo3(signature = (seed = None))]
	pub fn new(seed: Option<u64>) -> PyResult<Self> {
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

	// pickle
	fn __getnewargs__<'py>(
		&self,
		py: Python<'py>,
	) -> PyResult<Bound<'py, PyTuple>> {
		// not the actual seed, but the state will be restored by
		// `__setstate__`
		(0,).into_pyobject(py)
	}
}

py_pickle_state_impl!(PyRng, _pickle_impl);

pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_rng_rust_impl")?;

	m.add_class::<PyRng>()?;

	Ok(m)
}
