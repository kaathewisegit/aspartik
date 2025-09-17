#[cfg(feature = "python")]
use pyo3::prelude::*;

use core::f64;
use math::{
	consts::{LN_SQRT_2PI, SQRT_2PI},
	function::erf,
};
#[cfg(feature = "python")]
use util::impl_pyerr;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};

/// Implements the
/// [Log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{LogNormal, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = LogNormal::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), (0.5f64).exp());
/// assert_almost_eq!(n.pdf(1.0), 0.3989422804014327, epsilon = 1e-16);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct LogNormal {
	location: f64,
	scale: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for LogNormal;
	new(location: f64, scale: f64) throws LogNormalError;
	get(py_location) location: f64;
	get(py_scale) scale: f64;
	repr("LogNormal(location={}, scale={})", location, scale);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(location, scale);
}

/// Represents the errors that can occur when creating a [`LogNormal`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum LogNormalError {
	/// The location is NaN.
	LocationInvalid,

	/// The scale is NaN, zero or less than zero.
	ScaleInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(LogNormalError, pyo3::exceptions::PyValueError);

impl core::fmt::Display for LogNormalError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			LogNormalError::LocationInvalid => {
				write!(f, "Location is NaN")
			}
			LogNormalError::ScaleInvalid => write!(
				f,
				"Scale is NaN, zero or less than zero"
			),
		}
	}
}

impl std::error::Error for LogNormalError {}

impl LogNormal {
	/// Constructs a new log-normal distribution with a location of `location`
	/// and a scale of `scale`
	///
	/// # Errors
	///
	/// Returns an error if `location` or `scale` are `NaN`.
	/// Returns an error if `scale <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::LogNormal;
	///
	/// let mut result = LogNormal::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = LogNormal::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		location: f64,
		scale: f64,
	) -> Result<LogNormal, LogNormalError> {
		if location.is_nan() {
			return Err(LogNormalError::LocationInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(LogNormalError::ScaleInvalid);
		}

		Ok(LogNormal { location, scale })
	}

	/// Returns the location of the log-normal distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::LogNormal;
	///
	/// let n = LogNormal::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.location(), 0.0);
	/// ```
	pub fn location(&self) -> f64 {
		self.location
	}

	/// Returns the scale of the log-normal distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::LogNormal;
	///
	/// let n = LogNormal::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}
}

impl core::fmt::Display for LogNormal {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "LogNormal({}, {}^2)", self.location, self.scale)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for LogNormal {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		super::normal::sample_unchecked(rng, self.location, self.scale)
			.exp()
	}
}

impl ContinuousCDF for LogNormal {
	/// Calculates the cumulative distribution function for the log-normal
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) + (1 / 2) * erf((ln(x) - μ) / sqrt(2) * σ)
	/// ```
	///
	/// where `μ` is the location, `σ` is the scale, and `erf` is the
	/// error function
	fn cdf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			0.0
		} else if x.is_infinite() {
			1.0
		} else {
			0.5 * erf::erfc(
				(self.location - x.ln())
					/ (self.scale * f64::consts::SQRT_2),
			)
		}
	}

	/// Calculates the survival function for the log-normal
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) + (1 / 2) * erf(-(ln(x) - μ) / sqrt(2) * σ)
	/// ```
	///
	/// where `μ` is the location, `σ` is the scale, and `erf` is the
	/// error function
	///
	/// note that this calculates the complement due to flipping
	/// the sign of the argument error function with respect to the cdf.
	///
	/// the normal cdf Φ (and internal error function) as the following property:
	/// ```text
	///  Φ(-x) + Φ(x) = 1
	///  Φ(-x)        = 1 - Φ(x)
	/// ```
	fn sf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			1.0
		} else if x.is_infinite() {
			0.0
		} else {
			0.5 * erf::erfc(
				(x.ln() - self.location)
					/ (self.scale * f64::consts::SQRT_2),
			)
		}
	}

	/// Calculates the inverse cumulative distribution function for the
	/// log-normal distribution at `p`
	///
	/// # Panics
	///
	/// If `p < 0.0` or `p > 1.0`
	///
	/// # Formula
	///
	/// ```text
	/// μ - σ * sqrt(2) * erfc_inv(2p)
	/// ```
	///
	/// where `μ` is the location, `σ` is the scale and `erfc_inv` is
	/// the inverse of the complementary error function
	fn inverse_cdf(&self, p: f64) -> f64 {
		if p == 0.0 {
			0.0
		} else if p < 1.0 {
			(self.location
				- (self.scale
					* f64::consts::SQRT_2 * erf::erfc_inv(
					2.0 * p,
				)))
			.exp()
		} else if p == 1.0 {
			f64::INFINITY
		} else {
			panic!("p must be within [0.0, 1.0]");
		}
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for LogNormal {
	/// Returns the mean of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// e^(μ + σ^2 / 2)
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn mean(&self) -> Option<f64> {
		Some((self.location + self.scale * self.scale / 2.0).exp())
	}

	/// Returns the median of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// e^μ
	/// ```
	///
	/// where `μ` is the location
	fn median(&self) -> Option<f64> {
		Some(self.location.exp())
	}

	/// Returns the variance of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// (e^(σ^2) - 1) * e^(2μ + σ^2)
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn variance(&self) -> Option<f64> {
		let sigma2 = self.scale * self.scale;
		Some((sigma2.exp() - 1.0)
			* (self.location + self.location + sigma2).exp())
	}

	/// Returns the entropy of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(σe^(μ + 1 / 2) * sqrt(2π))
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn entropy(&self) -> Option<f64> {
		Some(0.5 + self.scale.ln() + self.location + LN_SQRT_2PI)
	}

	/// Returns the skewness of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// (e^(σ^2) + 2) * sqrt(e^(σ^2) - 1)
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn skewness(&self) -> Option<f64> {
		let expsigma2 = (self.scale * self.scale).exp();
		Some((expsigma2 + 2.0) * (expsigma2 - 1.0).sqrt())
	}
}

impl Mode<Option<f64>> for LogNormal {
	/// Returns the mode of the log-normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// e^(μ - σ^2)
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn mode(&self) -> Option<f64> {
		Some((self.location - self.scale * self.scale).exp())
	}
}

impl Continuous for LogNormal {
	type T = f64;

	/// Calculates the probability density function for the log-normal
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / xσ * sqrt(2π)) * e^(-((ln(x) - μ)^2) / 2σ^2)
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn pdf(&self, x: f64) -> f64 {
		if x <= 0.0 || x.is_infinite() {
			0.0
		} else {
			let d = (x.ln() - self.location) / self.scale;
			(-0.5 * d * d).exp() / (x * SQRT_2PI * self.scale)
		}
	}

	/// Calculates the log probability density function for the log-normal
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((1 / xσ * sqrt(2π)) * e^(-((ln(x) - μ)^2) / 2σ^2))
	/// ```
	///
	/// where `μ` is the location and `σ` is the scale
	fn ln_pdf(&self, x: f64) -> f64 {
		if x <= 0.0 || x.is_infinite() {
			f64::NEG_INFINITY
		} else {
			let d = (x.ln() - self.location) / self.scale;
			(-0.5 * d * d) - LN_SQRT_2PI - (x * self.scale).ln()
		}
	}
}
