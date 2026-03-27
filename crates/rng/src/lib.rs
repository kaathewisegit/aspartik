use anyhow::{Result, ensure};
use math::Probability;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use rand::{
	RngExt, SeedableRng,
	distr::uniform::{UniformFloat, UniformSampler},
};
use rand_pcg::Pcg64;

use std::io::Write;

use util::seconds_since_unix;

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

	pub fn dump(&self, mut dst: &mut dyn Write) -> Result<()> {
		fn write_u128<W: Write>(v: u128, dst: &mut W) -> Result<()> {
			let [upper, lower] = [(v >> 64) as u64, v as u64];
			rmp::encode::write_u64(dst, upper)?;
			rmp::encode::write_u64(dst, lower)?;
			Ok(())
		}

		let (state, increment) = self.inner().to_state();
		write_u128(state, &mut dst)?;
		write_u128(increment, &mut dst)?;
		Ok(())
	}

	pub fn load(&self, mut bytes: &[u8]) -> Result<()> {
		fn read_u128(src: &mut &[u8]) -> Result<u128> {
			let upper = rmp::decode::read_u64(src)?;
			let lower = rmp::decode::read_u64(src)?;
			Ok(((upper as u128) << 64) + lower as u128)
		}

		let state = read_u128(&mut bytes)?;
		let increment = read_u128(&mut bytes)?;
		let inner = &mut *self.inner();
		*inner = Pcg64::from_state(state, increment);
		Ok(())
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
		// This is a guessable, but `RNG` should be treated as a source
		// of cryptographic randomness anyways.
		let seed = seed.unwrap_or_else(seconds_since_unix);

		let inner = Pcg64::seed_from_u64(seed);

		Ok(PyRng {
			inner: Mutex::new(inner),
		})
	}

	/// Returns `true` with the probability of `ratio`
	///
	/// ## Exceptions
	///
	/// Will throw an exception if `ratio` is not in `[0, 1]`.
	#[pyo3(signature = (ratio = Probability::new(0.5)))]
	fn random_bool(&self, ratio: Probability<f64>) -> bool {
		self.inner().random_bool(*ratio)
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
