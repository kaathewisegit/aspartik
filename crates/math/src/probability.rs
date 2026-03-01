use num_traits::Float;
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};
use util::py_bail;

use std::ops::Deref;

/// A probability value which is always in the `[0, 1]` interval
///
/// It can be returned from a function or required as an argument to signal that
/// the value must be between 0 and 1, inclusively.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Probability<F>(F);

impl<F: Float> Probability<F> {
	pub fn new(p: F) -> Option<Self> {
		if p >= F::zero() && p <= F::one() {
			Some(Probability(p))
		} else {
			None
		}
	}

	/// Copies the inner value
	pub fn inner(&self) -> F {
		self.0
	}

	/// Consuming version of [`inner`][Probability::inner]
	///
	/// This function can be used to avoid copying.
	pub fn into_inner(self) -> F {
		self.0
	}
}

impl Probability<f64> {
	/// `const` constructor
	///
	/// It has to be implemented for a specific type because Rust currently
	/// doesn't allow const trait methods, and for a generic type comparison
	/// will use the `Ord` trait.
	pub const fn new_const(p: f64) -> Option<Self> {
		if p >= 0.0 && p <= 1.0 {
			Some(Probability(p))
		} else {
			None
		}
	}
}

impl<F> Deref for Probability<F> {
	type Target = F;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl From<Probability<f32>> for f32 {
	fn from(value: Probability<f32>) -> Self {
		value.into_inner()
	}
}
impl From<Probability<f64>> for f64 {
	fn from(value: Probability<f64>) -> Self {
		value.into_inner()
	}
}

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for Probability<f64> {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		let float = obj.extract::<f64>()?;

		let Some(out) = Probability::new(float) else {
			py_bail!(
				PyValueError,
				"Probability must be in [0, 1], got {float}"
			);
		};

		Ok(out)
	}
}
