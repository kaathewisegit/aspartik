#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};
use util::py_bail;

use std::{cmp::PartialOrd, ops::Deref};

/// A probability value which is always in the `[0, 1]` interval
///
/// It can be returned from a function or required as an argument to signal that
/// the value must be between 0 and 1, inclusively.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Probability<F>(F);

impl<F: Copy> Probability<F> {
	/// Copies the inner value
	pub fn inner(&self) -> F {
		self.0
	}

	/// Consuming version of [`inner`][Probability::inner]
	pub fn into_inner(self) -> F {
		self.0
	}
}

macro_rules! impls {
	($f:ty) => {
		impl Probability<$f> {
			pub const fn new(p: $f) -> Option<Self> {
				if 0.0 <= p && p <= 1.0 {
					Some(Probability(p))
				} else {
					None
				}
			}
		}

		impl From<Probability<$f>> for $f {
			fn from(value: Probability<$f>) -> Self {
				value.into_inner()
			}
		}
	};
}
impls!(f64);

impl<F> Deref for Probability<F> {
	type Target = F;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for Probability<f64> {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		let float = obj.extract::<f64>()?;

		let Some(out) = Probability::<f64>::new(float) else {
			py_bail!(
				PyValueError,
				"Probability must be in [0, 1], got {float}"
			);
		};

		Ok(out)
	}
}
