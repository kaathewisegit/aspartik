use pyo3::prelude::*;
use pyo3::{exceptions::PyTypeError, intern, types::PyString};
use rand::{rngs::OsRng, Rng as _, SeedableRng, TryRngCore};
use rand_pcg::Pcg64;

use std::sync::{Arc, Mutex, MutexGuard};

// TODO: add the compiler version here, probably via a build.rs script
const ABI: &str = concat!(env!("CARGO_PKG_VERSION"));

pub type Rng = Pcg64;

#[derive(Debug, Clone)]
#[pyclass(name = "Rng", module = "rng", frozen)]
#[repr(transparent)]
pub struct PyRng {
	inner: Arc<Mutex<Rng>>,
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

	pub fn downcast(any: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
		let name = any.get_type().fully_qualified_name()?;
		if name == "rng.Rng" {
			let py = any.py();
			let abi = any.getattr(intern!(py, "__abi"))?;
			let abi = abi.downcast::<PyString>()?;
			if abi != ABI {
				return Err(PyTypeError::new_err(format!("Wrong ABI version.  Expected {ABI}, got {abi}")));
			}

			// (un)SAFETY: this is not actually safe in the general
			// case.  Technically speaking, Rust reserves the right
			// to generate new type layout for each individual
			// *compilation*.  In practice I don't think that this
			// actually happens, so the two main concerns are the
			// crate versions and the Rust compiler versions.  The
			// `__abi` class attribute should have enough (note:
			// currently it doesn't) information to ensure that the
			// layout is the same.
			Ok(unsafe { any.downcast_into_unchecked::<PyRng>() })
		} else {
			Err(PyTypeError::new_err(format!(
				"Expected `PyRng`, got {name}",
			)))
		}
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
			inner: Arc::new(Mutex::new(inner)),
		})
	}

	#[classattr]
	fn __abi(py: Python<'_>) -> Bound<'_, PyString> {
		intern!(py, ABI).clone()
	}

	#[pyo3(signature = (ratio = 0.5))]
	fn random_bool(&self, ratio: f64) -> bool {
		self.inner().random_bool(ratio)
	}

	fn random_ratio(&self, numerator: u32, denominator: u32) -> bool {
		self.inner().random_ratio(numerator, denominator)
	}

	fn random_int(&self, lower: i64, upper: i64) -> i64 {
		self.inner().random_range(lower..upper)
	}
}

#[pymodule(name = "_rng_rust_impl")]
fn pymodule(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
	m.add_class::<PyRng>()?;

	Ok(())
}
