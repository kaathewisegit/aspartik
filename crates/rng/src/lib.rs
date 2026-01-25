use anyhow::{Result, ensure};
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use rand::{
	Rng as _, SeedableRng, TryRngCore,
	distr::uniform::{UniformFloat, UniformSampler},
	rngs::SysRng,
};
use rand_pcg::Pcg64;

use util::py_pickle_state_impl;

pub type Rng = Pcg64;

/// Random number generator
///
/// It is backed by a PCG64 random number generator implemented in Rust.  Its
/// output is a part of the stability guarantee: for the same seed and the same
/// minor version of `b3` it will always output the same stream of values.
///
/// It has a number of built-in methods, but it is primarily used by other
/// Aspartik modules (for example `b3` and `stats.distributions`) as a
/// randomness source.
///
///
/// ## Pickling
///
/// This class supports pickling.  It can be saved and then restored without
/// losing its internal state.
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
	/// New generator with a given seed
	///
	/// The seed must be positive and less than `2^64 - 1`.  If the seed is
	/// `None`, it'll be generated randomly.
	#[new]
	#[pyo3(signature = (seed = None))]
	pub fn new(seed: Option<u64>) -> PyResult<Self> {
		let seed =
			seed.unwrap_or_else(|| SysRng.try_next_u64().unwrap());

		let inner = Pcg64::seed_from_u64(seed);

		Ok(PyRng {
			inner: Mutex::new(inner),
		})
	}

	/// Returns `true` with the probability of `ratio`
	///
	///
	/// ## Exceptions
	///
	/// Will throw an exception if `ratio` is not in `[0, 1]`.
	#[pyo3(signature = (ratio = 0.5))]
	fn random_bool(&self, ratio: f64) -> bool {
		self.inner().random_bool(ratio)
	}

	/// Returns a random integer in `[lower, upper)`
	///
	/// `lower` and `higher` must be between `-2^63` and `2^63 - 1`.
	///
	///
	/// ## Exceptions
	///
	/// - If the arguments are out of bounds.
	/// - `lower >= upper`.
	fn random_int(&self, lower: i64, upper: i64) -> i64 {
		self.inner().random_range(lower..upper)
	}

	/// Returns a random floating point number in `[lower, upper]`
	///
	///
	/// ## Exceptions
	///
	/// - Throws an exception if `lower < upper` isn't true.  This means
	///   that neither `lower` nor `upper` can be `NaN`.
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
}

py_pickle_state_impl!(PyRng, _pickle_impl);

#[pymodule(name = "_rng_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use PyRng;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
