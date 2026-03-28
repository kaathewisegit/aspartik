#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

use core::f64::consts;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};
use math::{
	Probability,
	consts::{LN_SQRT_2PI, LN_SQRT_2PIE, SQRT_2PI},
	function::erf::{erfc, erfc_inv},
};
#[cfg(feature = "python")]
use util::impl_pyerr;

/// [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
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
	new(mean: f64, std_dev: f64) -> Result<Normal, NormalError>;
	repr("Normal(mean={}, std_dev={})", mean, std_dev);
	Continuous true;
	ContinuousCDF true;
	Distribution true;
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
	/// Constructs a normal distribution with a mean of `mean` (`μ`) and a
	/// standard deviation of `std_dev` (`σ`)
	///
	/// # Errors
	///
	/// Returns an error if `mean` or `std_dev` are `NaN` or if `std_dev <=
	/// 0.0`
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

	/// Normal distribution with a mean of 0 and a standard deviation of 1.
	pub fn standard() -> Normal {
		Normal {
			mean: 0.0,
			std_dev: 1.0,
		}
	}
}

impl core::fmt::Display for Normal {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Normal({}, {})", self.mean, self.std_dev)
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
	/// `½ · (1 + erf((x - μ) / (σ · sqrt(2))))`
	fn cdf(&self, x: f64) -> f64 {
		cdf_unchecked(x, self.mean, self.std_dev)
	}

	/// `½ · (1 + erf(-(x - μ) / (σ · sqrt(2))))`
	///
	/// Note that this calculates the complement due to flipping the sign of
	/// the argument error function with respect to the cdf.
	///
	/// the normal cdf Φ (and internal error function) as the following
	/// property:
	/// ```text
	///  Φ(-x) + Φ(x) = 1
	///  Φ(-x)        = 1 - Φ(x)
	/// ```
	fn sf(&self, x: f64) -> f64 {
		sf_unchecked(x, self.mean, self.std_dev)
	}

	/// `μ - sqrt(2) · σ · erfc_inv(2x)`
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;
		self.mean - (self.std_dev * consts::SQRT_2 * erfc_inv(2.0 * p))
	}

	/// `-∞`
	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	/// `+∞`
	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Normal {
	/// The mean of the normal distribution (`μ`)
	fn mean(&self) -> Option<f64> {
		Some(self.mean)
	}

	/// Equals the mean (`μ`)
	fn median(&self) -> Option<f64> {
		Some(self.mean)
	}

	/// `σ^2`
	fn variance(&self) -> Option<f64> {
		Some(self.std_dev * self.std_dev)
	}

	/// The standard deviation of the normal distribution (`σ`)
	fn std_dev(&self) -> Option<f64> {
		Some(self.std_dev)
	}

	/// `(1 / 2) * ln(2σ^2 * π * e)`
	fn entropy(&self) -> Option<f64> {
		Some(self.std_dev.ln() + LN_SQRT_2PIE)
	}

	/// Always zero
	fn skewness(&self) -> Option<f64> {
		Some(0.0)
	}
}

impl Mode<Option<f64>> for Normal {
	/// Always equals the mean (`μ`)
	fn mode(&self) -> Option<f64> {
		Some(self.mean)
	}
}

impl Continuous for Normal {
	/// `(1 / sqrt(2σ^2 · π)) · e^(-(x - μ)^2 / 2σ^2)`
	fn pdf(&self, x: f64) -> f64 {
		let d = (x - self.mean) / self.std_dev;
		(-0.5 * d * d).exp() / (SQRT_2PI * self.std_dev)
	}

	/// `ln((1 / sqrt(2σ^2 · π)) · e^(-(x - μ)^2 / 2σ^2))`
	fn ln_pdf(&self, x: f64) -> f64 {
		ln_pdf_unchecked(x, self.mean, self.std_dev)
	}
}

impl core::default::Default for Normal {
	/// Returns the standard normal distribution
	fn default() -> Self {
		Self::standard()
	}
}

pub(crate) fn cdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	0.5 * erfc((mean - x) / (std_dev * consts::SQRT_2))
}

pub(crate) fn sf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	0.5 * erfc((x - mean) / (std_dev * consts::SQRT_2))
}

pub(crate) fn pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	let d = (x - mean) / std_dev;
	(-0.5 * d * d).exp() / (SQRT_2PI * std_dev)
}

pub(crate) fn ln_pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
	let d = (x - mean) / std_dev;
	(-0.5 * d * d) - LN_SQRT_2PI - std_dev.ln()
}

#[cfg(feature = "rand")]
pub(crate) fn sample_unchecked<R: rand::Rng + ?Sized>(
	rng: &mut R,
	mean: f64,
	std_dev: f64,
) -> f64 {
	use crate::distribution::ziggurat;

	mean + std_dev * ziggurat::sample_std_normal(rng)
}
