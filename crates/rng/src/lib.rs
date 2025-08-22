use anyhow::{Result, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::{
	Rng as _, SeedableRng, TryRngCore,
	distr::uniform::{UniformFloat, UniformSampler},
	rngs::OsRng,
};
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
	pub fn inner(&self) -> MutexGuard<'_, Pcg64> {
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

	#[pyo3(signature = (lower = 0.0, upper = 1.0))]
	fn random_float(&self, lower: f64, upper: f64) -> Result<f64> {
		Ok(if lower == 0.0 && upper == 1.0 {
			self.inner().random()
		} else {
			ensure!(
				lower <= upper,
				"`lower` must be less than `upper`, got {lower} > {upper}",
			);
			let d = UniformFloat::<f64>::new(lower, upper)?;
			d.sample(&mut self.inner())
		})
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
	use util::py_make_submodule;
	let m = py_make_submodule!(py, "_rng_rust_impl");

	m.add_class::<PyRng>()?;

	Ok(m)
}
