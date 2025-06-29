use core::f64;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use math::function::{factorial, gamma};

#[cfg(feature = "python")]
use crate::python_macros::{impl_pyerr, impl_pymethods};
use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::*,
};

/// Implements the [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Poisson, Discrete};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = Poisson::new(1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0);
/// assert_almost_eq!(n.pmf(1), 0.367879441171442, 1e-15);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Poisson {
	lambda: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Poisson;
	new(lambda_: f64) throws PoissonError;
	get(as lambda_) lambda: f64;
	repr("Poisson({})", lambda);
	Discrete;
	DiscreteCDF;
	Distribution;
	pickle(lambda);
}

/// Represents the errors that can occur when creating a [`Poisson`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum PoissonError {
	/// The lambda is NaN, zero or less than zero.
	LambdaInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(PoissonError, pyo3::exceptions::PyValueError);

impl core::fmt::Display for PoissonError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			PoissonError::LambdaInvalid => write!(
				f,
				"Lambda is NaN, zero or less than zero"
			),
		}
	}
}

impl std::error::Error for PoissonError {}

impl Poisson {
	/// Constructs a new poisson distribution with a rate (λ)
	/// of `lambda`
	///
	/// # Errors
	///
	/// Returns an error if `lambda` is `NaN` or `lambda <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Poisson;
	///
	/// let mut result = Poisson::new(1.0);
	/// assert!(result.is_ok());
	///
	/// result = Poisson::new(0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(lambda: f64) -> Result<Poisson, PoissonError> {
		if lambda.is_nan() || lambda <= 0.0 {
			Err(PoissonError::LambdaInvalid)
		} else {
			Ok(Poisson { lambda })
		}
	}

	/// Returns the rate (λ) of the poisson distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Poisson;
	///
	/// let n = Poisson::new(1.0).unwrap();
	/// assert_eq!(n.lambda(), 1.0);
	/// ```
	pub fn lambda(&self) -> f64 {
		self.lambda
	}
}

impl core::fmt::Display for Poisson {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Pois({})", self.lambda)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<u64> for Poisson {
	/// Generates one sample from the Poisson distribution either by
	/// Knuth's method if lambda < 30.0 or Rejection method PA by
	/// A. C. Atkinson from the Journal of the Royal Statistical Society
	/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
	/// otherwise
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
		sample_unchecked(rng, self.lambda) as u64
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Poisson {
	/// Generates one sample from the Poisson distribution either by
	/// Knuth's method if lambda < 30.0 or Rejection method PA by
	/// A. C. Atkinson from the Journal of the Royal Statistical Society
	/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
	/// otherwise
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		sample_unchecked(rng, self.lambda)
	}
}

impl DiscreteCDF for Poisson {
	/// `P(x + 1, λ)`, where `λ` is the rate and `P` is the upper
	/// regularized gamma function
	///
	/// # Panics
	///
	/// If `x <= 0`.
	fn cdf(&self, x: u64) -> f64 {
		// lambda must be `>= 0`
		gamma::gamma_ur(x as f64 + 1.0, self.lambda)
			.expect("x must be positive")
	}

	/// `P(x + 1, λ)`, where `λ` is the rate and `P` is the lower
	/// regularized gamma function
	fn sf(&self, x: u64) -> f64 {
		// XXX: panics?
		gamma::gamma_lr(x as f64 + 1.0, self.lambda).unwrap()
	}

	fn lower(&self) -> u64 {
		0
	}

	fn upper(&self) -> u64 {
		u64::MAX
	}
}

impl Distribution for Poisson {
	/// Returns the mean of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// λ
	/// ```
	///
	/// where `λ` is the rate
	fn mean(&self) -> Option<f64> {
		Some(self.lambda)
	}

	/// Returns the median of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// floor(λ + 1 / 3 - 0.02 / λ)
	/// ```
	///
	/// where `λ` is the rate
	fn median(&self) -> Option<f64> {
		Some((self.lambda + 1.0 / 3.0 - 0.02 / self.lambda).floor())
	}

	/// Returns the variance of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// λ
	/// ```
	///
	/// where `λ` is the rate
	fn variance(&self) -> Option<f64> {
		Some(self.lambda)
	}

	/// Returns the entropy of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * ln(2πeλ) - 1 / (12λ) - 1 / (24λ^2) - 19 / (360λ^3)
	/// ```
	///
	/// where `λ` is the rate
	fn entropy(&self) -> Option<f64> {
		Some(0.5 * (2.0
			* f64::consts::PI * f64::consts::E
			* self.lambda)
			.ln() - 1.0 / (12.0 * self.lambda)
			- 1.0 / (24.0 * self.lambda * self.lambda)
			- 19.0 / (360.0
				* self.lambda * self.lambda * self.lambda))
	}

	/// Returns the skewness of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// λ^(-1/2)
	/// ```
	///
	/// where `λ` is the rate
	fn skewness(&self) -> Option<f64> {
		Some(1.0 / self.lambda.sqrt())
	}
}

impl Mode<Option<u64>> for Poisson {
	/// Returns the mode of the poisson distribution
	///
	/// # Formula
	///
	/// ```text
	/// floor(λ)
	/// ```
	///
	/// where `λ` is the rate
	fn mode(&self) -> Option<u64> {
		Some(self.lambda.floor() as u64)
	}
}

impl Discrete for Poisson {
	type T = u64;

	/// Calculates the probability mass function for the poisson distribution at
	/// `x`
	///
	/// # Formula
	///
	/// ```text
	/// (λ^x * e^(-λ)) / x!
	/// ```
	///
	/// where `λ` is the rate
	fn pmf(&self, x: u64) -> f64 {
		(-self.lambda + x as f64 * self.lambda.ln()
			- factorial::ln_factorial(x))
		.exp()
	}

	/// Calculates the log probability mass function for the poisson
	/// distribution at
	/// `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((λ^x * e^(-λ)) / x!)
	/// ```
	///
	/// where `λ` is the rate
	fn ln_pmf(&self, x: u64) -> f64 {
		-self.lambda + x as f64 * self.lambda.ln()
			- factorial::ln_factorial(x)
	}
}

/// Generates one sample from the Poisson distribution either by
/// Knuth's method if lambda < 30.0 or Rejection method PA by
/// A. C. Atkinson from the Journal of the Royal Statistical Society
/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
/// otherwise
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub fn sample_unchecked<R: rand::Rng + ?Sized>(
	rng: &mut R,
	lambda: f64,
) -> f64 {
	if lambda < 30.0 {
		let limit = (-lambda).exp();
		let mut count = 0.0;
		let mut product: f64 = rng.random();
		while product >= limit {
			count += 1.0;
			product *= rng.random::<f64>();
		}
		count
	} else {
		let c = 0.767 - 3.36 / lambda;
		let beta = f64::consts::PI / (3.0 * lambda).sqrt();
		let alpha = beta * lambda;
		let k = c.ln() - lambda - beta.ln();

		loop {
			let u: f64 = rng.random();
			let x = (alpha - ((1.0 - u) / u).ln()) / beta;
			let n = (x + 0.5).floor();
			if n < 0.0 {
				continue;
			}

			let v: f64 = rng.random();
			let y = alpha - beta * x;
			let temp = 1.0 + y.exp();
			let lhs = y + (v / (temp * temp)).ln();
			let rhs = k + n * lambda.ln()
				- factorial::ln_factorial(n as u64);
			if lhs <= rhs {
				return n;
			}
		}
	}
}
