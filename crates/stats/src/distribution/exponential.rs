#[cfg(feature = "python")]
use pyo3::prelude::*;

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

/// Implements the
/// [Exp](https://en.wikipedia.org/wiki/Exp_distribution)
/// distribution and is a special case of the
/// [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [here](./struct.Gamma.html))
///
/// # Examples
///
/// ```
/// use stats::distribution::{Exp, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Exp::new(1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0);
/// assert_eq!(n.pdf(1.0), 0.36787944117144233);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Exp {
	rate: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Exp;
	new(rate: f64) throws ExpError;
	get(py_rate) rate: f64;
	repr("Exp(rate={})", rate);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(rate);
}

/// Represents the errors that can occur when creating a [`Exp`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum ExpError {
	/// The rate is NaN, zero or less than zero.
	RateInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(ExpError, pyo3::exceptions::PyValueError);

impl core::fmt::Display for ExpError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			ExpError::RateInvalid => {
				write!(f, "Rate is NaN, zero or less than zero")
			}
		}
	}
}

impl std::error::Error for ExpError {}

impl Exp {
	/// Constructs a new exponential distribution with a
	/// rate (λ) of `rate`.
	///
	/// # Errors
	///
	/// Returns an error if rate is `NaN` or `rate <= 0.0`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Exp;
	///
	/// let mut result = Exp::new(1.0);
	/// assert!(result.is_ok());
	///
	/// result = Exp::new(-1.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(rate: f64) -> Result<Exp, ExpError> {
		if rate.is_nan() || rate <= 0.0 {
			Err(ExpError::RateInvalid)
		} else {
			Ok(Exp { rate })
		}
	}

	/// Returns the rate of the exponential distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Exp;
	///
	/// let n = Exp::new(1.0).unwrap();
	/// assert_eq!(n.rate(), 1.0);
	/// ```
	pub fn rate(&self) -> f64 {
		self.rate
	}
}

impl core::fmt::Display for Exp {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Exp({})", self.rate)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Exp {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		use crate::distribution::ziggurat;

		ziggurat::sample_exp_1(r) / self.rate
	}
}

impl ContinuousCDF for Exp {
	/// Calculates the cumulative distribution function for the
	/// exponential distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - e^(-λ * x)
	/// ```
	///
	/// where `λ` is the rate
	fn cdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else {
			1.0 - (-self.rate * x).exp()
		}
	}

	/// Calculates the cumulative distribution function for the
	/// exponential distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// e^(-λ * x)
	/// ```
	///
	/// where `λ` is the rate
	fn sf(&self, x: f64) -> f64 {
		if x < 0.0 { 1.0 } else { (-self.rate * x).exp() }
	}

	/// `-ln(1 - p) / λ`, where `p` is the probability and `λ` is the rate.
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;

		-(-p).ln_1p() / self.rate
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Exp {
	/// Returns the mean of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1 / λ
	/// ```
	///
	/// where `λ` is the rate
	fn mean(&self) -> Option<f64> {
		Some(1.0 / self.rate)
	}

	/// Returns the median of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 / λ) * ln2
	/// ```
	///
	/// where `λ` is the rate
	fn median(&self) -> Option<f64> {
		Some(f64::consts::LN_2 / self.rate)
	}

	/// Returns the variance of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1 / λ^2
	/// ```
	///
	/// where `λ` is the rate
	fn variance(&self) -> Option<f64> {
		Some(1.0 / (self.rate * self.rate))
	}

	/// Returns the entropy of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1 - ln(λ)
	/// ```
	///
	/// where `λ` is the rate
	fn entropy(&self) -> Option<f64> {
		Some(1.0 - self.rate.ln())
	}

	/// Returns the skewness of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// 2
	/// ```
	fn skewness(&self) -> Option<f64> {
		Some(2.0)
	}
}

impl Mode<Option<f64>> for Exp {
	/// Returns the mode of the exponential distribution
	///
	/// # Formula
	///
	/// ```text
	/// 0
	/// ```
	fn mode(&self) -> Option<f64> {
		Some(0.0)
	}
}

impl Continuous for Exp {
	type T = f64;

	/// Calculates the probability density function for the exponential
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// λ * e^(-λ * x)
	/// ```
	///
	/// where `λ` is the rate
	fn pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else {
			self.rate * (-self.rate * x).exp()
		}
	}

	/// Calculates the log probability density function for the exponential
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln(λ * e^(-λ * x))
	/// ```
	///
	/// where `λ` is the rate
	fn ln_pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			f64::NEG_INFINITY
		} else {
			self.rate.ln() - self.rate * x
		}
	}
}
