#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::python_macros::{impl_pyerr, impl_pymethods};
use crate::{
	distribution::{Continuous, ContinuousCDF},
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
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Uniform {
	min: f64,
	max: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Uniform;
	new(min: f64, max: f64) throws UniformError;
	get(py_min) min: f64;
	get(py_max) max: f64;
	repr("Uniform(min={}, max={})", min, max);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(min, max);
}

/// Represents the errors that can occur when creating a [`Uniform`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum UniformError {
	/// The minimum is NaN or infinite.
	MinInvalid,

	/// The maximum is NaN or infinite.
	MaxInvalid,

	/// The maximum is not greater than the minimum.
	MaxNotGreaterThanMin,
}

#[cfg(feature = "python")]
impl_pyerr!(UniformError, pyo3::exceptions::PyValueError);

impl core::fmt::Display for UniformError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			UniformError::MinInvalid => {
				write!(f, "Minimum is NaN or infinite")
			}
			UniformError::MaxInvalid => {
				write!(f, "Maximum is NaN or infinite")
			}
			UniformError::MaxNotGreaterThanMin => {
				write!(
					f,
					"Maximum is not greater than the minimum"
				)
			}
		}
	}
}

impl std::error::Error for UniformError {}

impl Uniform {
	/// Constructs a new uniform distribution with a min of `min` and a max
	/// of `max`.
	///
	/// # Errors
	///
	/// Returns an error if `min` or `max` are `NaN` or infinite.
	/// Returns an error if `min >= max`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Uniform;
	///
	/// let mut result = Uniform::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Uniform::new(f64::NAN, f64::NAN);
	/// assert!(result.is_err());
	///
	/// result = Uniform::new(f64::NEG_INFINITY, 1.0);
	/// assert!(result.is_err());
	/// ```
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

	/// Constructs a new standard uniform distribution with
	/// a lower bound 0 and an upper bound of 1.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Uniform;
	///
	/// let uniform = Uniform::standard();
	/// ```
	pub fn standard() -> Self {
		Self { min: 0.0, max: 1.0 }
	}

	/// Returns the lower bound of the uniform distribution
	/// as a `f64`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Uniform;
	///
	/// let uniform = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(uniform.min(), 0.0);
	/// ```
	pub fn min(&self) -> f64 {
		self.min
	}

	/// Returns the upper bound of the uniform distribution
	/// as a `f64`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Uniform;
	///
	/// let uniform = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(uniform.max(), 1.0);
	/// ```
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
		write!(f, "Uni([{},{}])", self.min, self.max)
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

	/// Finds the value of `x` where `F(p) = x`
	fn inverse_cdf(&self, p: f64) -> f64 {
		if !(0.0..=1.0).contains(&p) {
			panic!("p must be in [0, 1], was {p}");
		} else if p == 0.0 {
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
	type T = f64;

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
