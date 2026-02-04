#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

use core::f64;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};
use math::{
	consts::{LN_SQRT_2PI, LN_SQRT_2PIE, SQRT_2PI},
	function::erf,
};
#[cfg(feature = "python")]
use util::impl_pyerr;

/// Implements the [Normal](https://en.wikipedia.org/wiki/Normal_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Normal, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Normal::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.24197072451914334);
/// ```
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
pub struct Normal {
	mean: f64,
	std_dev: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Normal;
	new(mean: f64, std_dev: f64) throws NormalError;
	repr("Normal(mean={}, std_dev={})", mean, std_dev);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(mean, std_dev);
}

/// Represents the errors that can occur when creating a [`Normal`].
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
pub enum NormalError {
	#[error("The mean is NaN")]
	MeanInvalid,

	#[error("The standard deviation is NaN, zero or less than zero")]
	StandardDeviationInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(NormalError, pyo3::exceptions::PyValueError);

impl Normal {
	///  Constructs a new normal distribution with a mean of `mean`
	/// and a standard deviation of `std_dev`
	///
	/// # Errors
	///
	/// Returns an error if `mean` or `std_dev` are `NaN` or if
	/// `std_dev <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Normal;
	///
	/// let mut result = Normal::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Normal::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(mean: f64, std_dev: f64) -> Result<Normal, NormalError> {
		if mean.is_nan() {
			return Err(NormalError::MeanInvalid);
		}

		if std_dev.is_nan() || std_dev <= 0.0 {
			return Err(NormalError::StandardDeviationInvalid);
		}

		Ok(Normal { mean, std_dev })
	}

	/// Constructs a new standard normal distribution with a mean of 0
	/// and a standard deviation of 1.
	///
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Normal;
	///
	/// let mut result = Normal::standard();
	/// ```
	pub fn standard() -> Normal {
		Normal {
			mean: 0.0,
			std_dev: 1.0,
		}
	}
}

impl core::fmt::Display for Normal {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "N({},{})", self.mean, self.std_dev)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Normal {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		sample_unchecked(rng, self.mean, self.std_dev)
	}
}

impl ContinuousCDF for Normal {
	/// Calculates the cumulative distribution function for the
	/// normal distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * (1 + erf((x - μ) / (σ * sqrt(2))))
	/// ```
	///
	/// where `μ` is the mean, `σ` is the standard deviation, and
	/// `erf` is the error function
	fn cdf(&self, x: f64) -> f64 {
		cdf_unchecked(x, self.mean, self.std_dev)
	}

	/// Calculates the survival function for the
	/// normal distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * (1 + erf(-(x - μ) / (σ * sqrt(2))))
	/// ```
	///
	/// where `μ` is the mean, `σ` is the standard deviation, and
	/// `erf` is the error function
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
		sf_unchecked(x, self.mean, self.std_dev)
	}

	/// `μ - sqrt(2) * σ * erfc_inv(2x)`, where `μ` is the mean, `σ` is the
	/// standard deviation and `erfc_inv` is the inverse of the
	/// complementary error function.
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;
		self.mean
			- (self.std_dev
				* f64::consts::SQRT_2 * erf::erfc_inv(2.0 * p))
	}

	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Normal {
	/// Returns the mean of the normal distribution
	///
	/// # Remarks
	///
	/// This is the same mean used to construct the distribution
	fn mean(&self) -> Option<f64> {
		Some(self.mean)
	}

	/// Returns the median of the normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the mean
	fn median(&self) -> Option<f64> {
		Some(self.mean)
	}

	/// Returns the variance of the normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// σ^2
	/// ```
	///
	/// where `σ` is the standard deviation
	fn variance(&self) -> Option<f64> {
		Some(self.std_dev * self.std_dev)
	}

	/// Returns the standard deviation of the normal distribution
	/// # Remarks
	/// This is the same standard deviation used to construct the distribution
	fn std_dev(&self) -> Option<f64> {
		Some(self.std_dev)
	}

	/// Returns the entropy of the normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * ln(2σ^2 * π * e)
	/// ```
	///
	/// where `σ` is the standard deviation
	fn entropy(&self) -> Option<f64> {
		Some(self.std_dev.ln() + LN_SQRT_2PIE)
	}

	/// Returns the skewness of the normal distribution
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

impl Mode<Option<f64>> for Normal {
	/// Returns the mode of the normal distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the mean
	fn mode(&self) -> Option<f64> {
		Some(self.mean)
	}
}

impl Continuous for Normal {
	/// Calculates the probability density function for the normal distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / sqrt(2σ^2 * π)) * e^(-(x - μ)^2 / 2σ^2)
	/// ```
	///
	/// where `μ` is the mean and `σ` is the standard deviation
	fn pdf(&self, x: f64) -> f64 {
		pdf_unchecked(x, self.mean, self.std_dev)
	}

	/// Calculates the log probability density function for the normal
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((1 / sqrt(2σ^2 * π)) * e^(-(x - μ)^2 / 2σ^2))
	/// ```
	///
	/// where `μ` is the mean and `σ` is the standard deviation
	fn ln_pdf(&self, x: f64) -> f64 {
		ln_pdf_unchecked(x, self.mean, self.std_dev)
	}
}

/// performs an unchecked cdf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn cdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	0.5 * erf::erfc((mean - x) / (std_dev * f64::consts::SQRT_2))
}

/// performs an unchecked sf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn sf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	0.5 * erf::erfc((x - mean) / (std_dev * f64::consts::SQRT_2))
}

/// performs an unchecked pdf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	let d = (x - mean) / std_dev;
	(-0.5 * d * d).exp() / (SQRT_2PI * std_dev)
}

/// performs an unchecked log(pdf) calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn ln_pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	let d = (x - mean) / std_dev;
	(-0.5 * d * d) - LN_SQRT_2PI - std_dev.ln()
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
/// draws a sample from a normal distribution using the Box-Muller algorithm
pub fn sample_unchecked<R: rand::Rng + ?Sized>(
	rng: &mut R,
	mean: f64,
	std_dev: f64,
) -> f64 {
	use crate::distribution::ziggurat;

	mean + std_dev * ziggurat::sample_std_normal(rng)
}

impl core::default::Default for Normal {
	/// Returns the standard normal distribution with a mean of 0
	/// and a standard deviation of 1.
	fn default() -> Self {
		Self::standard()
	}
}
