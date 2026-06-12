use computare_core::Float;
use computare_special::erf::{erfc, inverse_erfc};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

use std::f64::consts::SQRT_2;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};
#[cfg(feature = "python")]
use util::impl_pyerr;

/// Implements the
/// [Log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
/// distribution
#[derive(Copy, Clone, PartialEq, Debug)]
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
pub struct LogNormal {
	location: f64,
	scale: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for LogNormal;
	new(location: f64, scale: f64) -> Result<LogNormal, LogNormalError>;
	get(location: f64 as py_location);
	get(scale: f64 as py_scale);
	repr("LogNormal(location={}, scale={})", location, scale);
	Continuous true;
	ContinuousCDF true;
	Distribution true;
}

/// Represents the errors that can occur when creating a [`LogNormal`].
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
pub enum LogNormalError {
	#[error("The location is NaN")]
	LocationInvalid,

	#[error("The scale is NaN, zero or less than zero")]
	ScaleInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(LogNormalError, pyo3::exceptions::PyValueError);

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
			0.5 * erfc((self.location - x.ln())
				/ (self.scale * SQRT_2))
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
			0.5 * erfc((x.ln() - self.location)
				/ (self.scale * SQRT_2))
		}
	}

	/// `μ - σ * sqrt(2) * erfc_inv(2p)`, where `μ` is the location, `σ` is
	/// the scale and `erfc_inv` is the inverse of the complementary error
	/// function.
	fn inverse_cdf(&self, p: f64) -> f64 {
		if p == 0.0 {
			0.0
		} else if p == 1.0 {
			f64::INFINITY
		} else {
			(self.location
				- (self.scale * SQRT_2 * inverse_erfc(2.0 * p)))
			.exp()
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
		Some(0.5 + self.scale.ln() + self.location + f64::LN_SQRT_2PI)
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
			(-0.5 * d * d).exp() / (x * f64::SQRT_2PI * self.scale)
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
			(-0.5 * d * d)
				- f64::LN_SQRT_2PI - (x * self.scale).ln()
		}
	}
}
