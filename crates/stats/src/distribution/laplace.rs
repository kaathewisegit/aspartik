#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

use core::f64;
#[cfg(feature = "python")]
use util::impl_pyerr;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};

/// Implements the [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
/// distribution.
///
/// # Examples
///
/// ```
/// use stats::distribution::{Laplace, Continuous};
/// use stats::statistics::Mode;
///
/// let n = Laplace::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.18393972058572117);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Laplace {
	location: f64,
	scale: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Laplace;
	new(location: f64, scale: f64) throws LaplaceError;
	get(py_location) location: f64;
	get(py_scale) scale: f64;
	repr("Laplace(location={}, scale={})", location, scale);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(location, scale);
}

/// Represents the errors that can occur when creating a [`Laplace`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum LaplaceError {
	#[error("The location is NaN")]
	LocationInvalid,

	#[error("The scale is NaN, zero or less than zero")]
	ScaleInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(LaplaceError, pyo3::exceptions::PyValueError);

impl Laplace {
	/// Constructs a new laplace distribution with the given
	/// location and scale.
	///
	/// # Errors
	///
	/// Returns an error if location or scale are `NaN` or `scale <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Laplace;
	///
	/// let mut result = Laplace::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Laplace::new(0.0, -1.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(location: f64, scale: f64) -> Result<Laplace, LaplaceError> {
		if location.is_nan() {
			return Err(LaplaceError::LocationInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(LaplaceError::ScaleInvalid);
		}

		Ok(Laplace { location, scale })
	}

	/// Returns the location of the laplace distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Laplace;
	///
	/// let n = Laplace::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.location(), 0.0);
	/// ```
	pub fn location(&self) -> f64 {
		self.location
	}

	/// Returns the scale of the laplace distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Laplace;
	///
	/// let n = Laplace::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}
}

impl core::fmt::Display for Laplace {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Laplace({}, {})", self.location, self.scale)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Laplace {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let x: f64 = rng.random_range(-0.5..0.5);
		self.location
			- self.scale * x.signum() * (1.0 - 2.0 * x.abs()).ln()
	}
}

impl ContinuousCDF for Laplace {
	/// Calculates the cumulative distribution function for the
	/// laplace distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * (1 + signum(x - μ)) - signum(x - μ) * exp(-|x - μ| / b)
	/// ```
	///
	/// where `μ` is the location, `b` is the scale
	fn cdf(&self, x: f64) -> f64 {
		let y = (-(x - self.location).abs() / self.scale).exp() / 2.0;
		if x >= self.location { 1.0 - y } else { y }
	}

	/// Calculates the survival function for the
	/// laplace distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - [(1 / 2) * (1 + signum(x - μ)) - signum(x - μ) * exp(-|x - μ| / b)]
	/// ```
	///
	/// where `μ` is the location, `b` is the scale
	fn sf(&self, x: f64) -> f64 {
		let y = (-(x - self.location).abs() / self.scale).exp() / 2.0;
		if x >= self.location { y } else { 1.0 - y }
	}

	/// `μ + b * ln(2p)` if `p <= 1/2`, `μ - b * ln(2 - 2p)` otherwise,
	/// where `μ` is the location, `b` is the scale.
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;

		if p <= 0.5 {
			self.location + self.scale * (2.0 * p).ln()
		} else {
			self.location - self.scale * (2.0 - 2.0 * p).ln()
		}
	}

	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Laplace {
	/// Returns the mode of the laplace distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn mean(&self) -> Option<f64> {
		Some(self.location)
	}

	/// Returns the median of the laplace distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn median(&self) -> Option<f64> {
		Some(self.location)
	}

	/// Returns the variance of the laplace distribution
	///
	/// # Formula
	///
	/// ```text
	/// 2*b^2
	/// ```
	///
	/// where `b` is the scale
	fn variance(&self) -> Option<f64> {
		Some(2.0 * self.scale * self.scale)
	}

	/// Returns the entropy of the laplace distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(2be)
	/// ```
	///
	/// where `b` is the scale
	fn entropy(&self) -> Option<f64> {
		Some((2.0 * self.scale).ln() + 1.0)
	}

	/// Returns the skewness of the laplace distribution
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

impl Mode<Option<f64>> for Laplace {
	/// Returns the mode of the laplace distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn mode(&self) -> Option<f64> {
		Some(self.location)
	}
}

impl Continuous for Laplace {
	/// Calculates the probability density function for the laplace
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2b) * exp(-|x - μ| / b)
	/// ```
	/// where `μ` is the location and `b` is the scale
	fn pdf(&self, x: f64) -> f64 {
		(-(x - self.location).abs() / self.scale).exp()
			/ (2.0 * self.scale)
	}

	/// Calculates the log probability density function for the laplace
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((1 / 2b) * exp(-|x - μ| / b))
	/// ```
	///
	/// where `μ` is the location and `b` is the scale
	fn ln_pdf(&self, x: f64) -> f64 {
		((-(x - self.location).abs() / self.scale).exp()
			/ (2.0 * self.scale))
			.ln()
	}
}
