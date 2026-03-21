#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};
use util::py_bail;

use std::{cmp::PartialOrd, fmt, ops::Deref};

/// A probability value which is always in the `[0, 1]` interval
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Probability<F>(F);

/// A positive value which is strictly larger than zero
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Positive<F>(F);

macro_rules! common_impls {
	($wrapper:ident) => {
		impl<F: Copy> $wrapper<F> {
			/// Copies the inner value
			pub fn inner(&self) -> F {
				self.0
			}

			/// Consuming version of [`inner`][Probability::inner]
			pub fn into_inner(self) -> F {
				self.0
			}
		}

		impl<F> Deref for $wrapper<F> {
			type Target = F;

			fn deref(&self) -> &Self::Target {
				&self.0
			}
		}

		impl<F: fmt::Display> fmt::Display for $wrapper<F> {
			fn fmt(
				&self,
				f: &mut fmt::Formatter<'_>,
			) -> Result<(), fmt::Error> {
				self.0.fmt(f)
			}
		}
	};
}
common_impls!(Probability);
common_impls!(Positive);

macro_rules! float_impls {
	($wrapper:ident, $f:ty, $val:ident, $cond:expr, $range:literal) => {
		impl $wrapper<$f> {
			pub const fn new($val: $f) -> Option<Self> {
				if $cond { Some(Self($val)) } else { None }
			}
		}

		impl From<$wrapper<$f>> for $f {
			fn from(value: $wrapper<$f>) -> Self {
				value.into_inner()
			}
		}

		#[cfg(feature = "python")]
		impl<'py> FromPyObject<'_, 'py> for $wrapper<$f> {
			type Error = PyErr;

			fn extract(
				obj: Borrowed<'_, 'py, PyAny>,
			) -> PyResult<Self> {
				let float = obj.extract::<f64>()?;

				let Some(out) = $wrapper::new(float) else {
					py_bail!(
						PyValueError,
						"{} must be in {}, got {}",
						stringify!($wrapper),
						$range,
						float,
					);
				};

				Ok(out)
			}
		}

		#[cfg(feature = "python")]
		impl<'py> IntoPyObject<'py> for $wrapper<$f> {
			type Target = pyo3::types::PyFloat;
			type Output = Bound<'py, pyo3::types::PyFloat>;
			type Error = PyErr;

			fn into_pyobject(
				self,
				py: Python<'py>,
			) -> Result<Self::Output, Self::Error> {
				Ok(self.0.into_pyobject(py)?)
			}
		}
	};
}

float_impls!(Probability, f64, p, 0.0 <= p && p <= 1.0, "[0, 1]");
float_impls!(Positive, f64, p, 0.0 < p, "(0, inf)");
