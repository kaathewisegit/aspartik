#[cfg(feature = "python")]
use pyo3::prelude::*;

use core::f64::consts;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};

/// Implements the [Exp](https://en.wikipedia.org/wiki/Exp_distribution)
/// distribution and is a special case of the
/// [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [here](./struct.Gamma.html))
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
pub struct Exp {
	rate: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Exp;
	new(rate: f64) -> Exp;
	get(rate: f64 as py_rate);
	repr("Exp(rate={})", rate);
	Continuous true;
	ContinuousCDF true;
	Distribution true;
}

impl Exp {
	/// Constructs a new exponential distribution with a rate (λ) of `rate`.
	pub fn new(rate: f64) -> Exp {
		Exp { rate }
	}

	/// Returns the rate of the exponential distribution
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
	fn inverse_cdf(&self, p: f64) -> f64 {
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
		Some(consts::LN_2 / self.rate)
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
