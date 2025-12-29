use num_traits::Float;

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
