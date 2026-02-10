#[cfg(feature = "python")]
use pyo3::prelude::*;
use rand::RngExt;
use thiserror::Error;

#[cfg(feature = "python")]
use util::impl_pyerr;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};

/// Implements the [Continuous
/// Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Uniform, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Uniform::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.5);
/// assert_eq!(n.pdf(0.5), 1.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(
	feature = "python",
	pyclass(
		from_py_object,
		module = "aspartik.stats.distributions",
		frozen,
		eq,
		str
	)
)]
pub struct Uniform {
	min: f64,
	max: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Uniform;
	new(min: f64, max: f64) throws UniformError;
	get(min: f64 as py_min);
	get(max: f64 as py_max);
	repr("Uniform(min={}, max={})", min, max);
	Continuous true;
	ContinuousCDF true;
	Distribution true;
}

/// Represents the errors that can occur when creating a [`Uniform`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(
		from_py_object,
		module = "aspartik.stats.distributions",
		frozen,
		eq,
		str
	)
)]
pub enum UniformError {
	#[error("The minimum is NaN or infinite")]
	MinInvalid,

	#[error("The maximum is NaN or infinite")]
	MaxInvalid,

	#[error("The maximum is not greater than the minimum")]
	MaxNotGreaterThanMin,
}

#[cfg(feature = "python")]
impl_pyerr!(UniformError, pyo3::exceptions::PyValueError);

impl Uniform {
	/// A uniform distribution on `[min, max]`
	///
	/// # Errors
	///
	/// - If `min` or `max` are `NaN` or infinite.
	/// - If `min >= max`.
	pub fn new(min: f64, max: f64) -> Result<Uniform, UniformError> {
		if !min.is_finite() {
			return Err(UniformError::MinInvalid);
		}

		if !max.is_finite() {
			return Err(UniformError::MaxInvalid);
		}

		if min < max {
			Ok(Uniform { min, max })
		} else {
			Err(UniformError::MaxNotGreaterThanMin)
		}
	}

	/// A uniform distribution on `[0, 1]`
	pub fn standard() -> Self {
		Self { min: 0.0, max: 1.0 }
	}

	/// Lower bound
	pub fn min(&self) -> f64 {
		self.min
	}

	/// Upper bound
	pub fn max(&self) -> f64 {
		self.max
	}
}

impl Default for Uniform {
	fn default() -> Self {
		Self::standard()
	}
}

impl core::fmt::Display for Uniform {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Uniform([{},{}])", self.min, self.max)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Uniform {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let d = rand::distr::Uniform::new_inclusive(self.min, self.max)
			.unwrap();
		rng.sample(d)
	}
}

impl ContinuousCDF for Uniform {
	/// Calculates the cumulative distribution function for the uniform
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (x - min) / (max - min)
	/// ```
	fn cdf(&self, x: f64) -> f64 {
		if x <= self.min {
			0.0
		} else if x >= self.max {
			1.0
		} else {
			(x - self.min) / (self.max - self.min)
		}
	}

	/// Calculates the survival function for the uniform
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (max - x) / (max - min)
	/// ```
	fn sf(&self, x: f64) -> f64 {
		if x <= self.min {
			1.0
		} else if x >= self.max {
			0.0
		} else {
			(self.max - x) / (self.max - self.min)
		}
	}

	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;

		if p == 0.0 {
			self.min
		} else if p == 1.0 {
			self.max
		} else {
			(self.max - self.min) * p + self.min
		}
	}

	fn lower(&self) -> f64 {
		self.min
	}

	fn upper(&self) -> f64 {
		self.max
	}
}

impl Distribution for Uniform {
	/// Returns the mean for the continuous uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// (min + max) / 2
	/// ```
	fn mean(&self) -> Option<f64> {
		Some((self.min + self.max) / 2.0)
	}

	/// Returns the median for the continuous uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// (min + max) / 2
	/// ```
	fn median(&self) -> Option<f64> {
		Some((self.min + self.max) / 2.0)
	}

	/// Returns the variance for the continuous uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// (max - min)^2 / 12
	/// ```
	fn variance(&self) -> Option<f64> {
		Some((self.max - self.min) * (self.max - self.min) / 12.0)
	}

	/// Returns the entropy for the continuous uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(max - min)
	/// ```
	fn entropy(&self) -> Option<f64> {
		Some((self.max - self.min).ln())
	}

	/// Returns the skewness for the continuous uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// 0
	/// ```
	fn skewness(&self) -> Option<f64> {
		Some(0.0)
	}
}

impl Mode<Option<f64>> for Uniform {
	/// Returns the mode for the continuous uniform distribution
	///
	/// # Remarks
	///
	/// Since every element has an equal probability, mode simply
	/// returns the middle element
	///
	/// # Formula
	///
	/// ```text
	/// N/A // (max + min) / 2 for the middle element
	/// ```
	fn mode(&self) -> Option<f64> {
		Some((self.min + self.max) / 2.0)
	}
}

impl Continuous for Uniform {
	/// Calculates the probability density function for the continuous uniform
	/// distribution at `x`
	///
	/// # Remarks
	///
	/// Returns `0.0` if `x` is not in `[min, max]`
	///
	/// # Formula
	///
	/// ```text
	/// 1 / (max - min)
	/// ```
	fn pdf(&self, x: f64) -> f64 {
		if x < self.min || x > self.max {
			0.0
		} else {
			1.0 / (self.max - self.min)
		}
	}

	/// Calculates the log probability density function for the continuous
	/// uniform
	/// distribution at `x`
	///
	/// # Remarks
	///
	/// Returns `f64::NEG_INFINITY` if `x` is not in `[min, max]`
	///
	/// # Formula
	///
	/// ```text
	/// ln(1 / (max - min))
	/// ```
	fn ln_pdf(&self, x: f64) -> f64 {
		if x < self.min || x > self.max {
			f64::NEG_INFINITY
		} else {
			-(self.max - self.min).ln()
		}
	}
}
