use approx::ulps_eq;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::python_macros::{impl_pyerr, impl_pymethods};
use crate::{
	distribution::{Continuous, ContinuousCDF},
	function::{beta, gamma},
	statistics::{Distribution, Mode},
};

/// [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
///
/// # Examples
///
/// ```
/// use stats::distribution::{Beta, Continuous};
/// use stats::statistics::*;
/// use math::assert_almost_eq;
///
/// let n = Beta::new(2.0, 2.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.5);
/// assert_almost_eq!(n.pdf(0.5), 1.5, 1e-14);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Beta {
	shape_a: f64,
	shape_b: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Beta;
	new(shape_a: f64, shape_b: f64) throws BetaError;
	get(py_shape_a) shape_a: f64;
	get(py_shape_b) shape_b: f64;
	repr("Beta(shape_a={}, shape_b={})", shape_a, shape_b);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(shape_a, shape_b);
}

/// Represents the errors that can occur when creating a [`Beta`].
#[derive(Copy, Clone, PartialEq, Debug, Error)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum BetaError {
	/// Shape α is NaN, infinite, zero or negative
	#[error("Shape α must be a positive non-zero finite value")]
	InvalidAlpha,

	/// Shape β is NaN, infinite, zero or negative
	#[error("Shape β must be a positive non-zero finite value")]
	InvalidBeta,
}

#[cfg(feature = "python")]
impl_pyerr!(BetaError, pyo3::exceptions::PyValueError);

impl Beta {
	/// Constructs a new beta distribution with `shape_a` as alpha (α) and
	/// and `shape_b` as beta (β).
	///
	/// Both α and β must be positive, non-zero, and finite.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Beta, BetaError};
	///
	/// assert!(Beta::new(2.0, 2.0).is_ok());
	/// assert_eq!(Beta::new(0.0, 1.0), Err(BetaError::InvalidAlpha));
	/// ```
	pub fn new(shape_a: f64, shape_b: f64) -> Result<Beta, BetaError> {
		if shape_a.is_nan() || shape_a.is_infinite() || shape_a <= 0.0 {
			return Err(BetaError::InvalidAlpha);
		}

		if shape_b.is_nan() || shape_b.is_infinite() || shape_b <= 0.0 {
			return Err(BetaError::InvalidBeta);
		}

		Ok(Beta { shape_a, shape_b })
	}

	/// The alpha parameter (α)
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Beta;
	///
	/// let n = Beta::new(1.0, 2.0).unwrap();
	/// assert_eq!(n.shape_a(), 1.0);
	/// ```
	pub fn shape_a(&self) -> f64 {
		self.shape_a
	}

	/// The beta parameter (β)
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Beta;
	///
	/// let n = Beta::new(1.0, 2.0).unwrap();
	/// assert_eq!(n.shape_b(), 2.0);
	/// ```
	pub fn shape_b(&self) -> f64 {
		self.shape_b
	}
}

impl core::fmt::Display for Beta {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Beta(a={}, b={})", self.shape_a, self.shape_b)
	}
}

impl ContinuousCDF for Beta {
	/// `I_x(α, β)`, where `I_x` is the regularized lower incomplete beta
	/// function.
	fn cdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else if x >= 1.0 {
			1.0
		} else if ulps_eq!(self.shape_a, 1.0)
			&& ulps_eq!(self.shape_b, 1.0)
		{
			x
		} else {
			beta::beta_reg(self.shape_a, self.shape_b, x)
		}
	}

	/// `I_(1-x)(β, α)`, where `I_x` is the regularized lower incomplete
	/// beta function.
	fn sf(&self, x: f64) -> f64 {
		if x < 0.0 {
			1.0
		} else if x >= 1.0 {
			0.0
		} else if ulps_eq!(self.shape_a, 1.0)
			&& ulps_eq!(self.shape_b, 1.0)
		{
			1.0 - x
		} else {
			beta::beta_reg(self.shape_b, self.shape_a, 1.0 - x)
		}
	}

	/// `I_x^{-1}(β, α)`, where `I_x` is the regularized lower incomplete
	/// beta function.
	///
	/// # Panics
	///
	/// If x is not in `[0, 1]`.
	fn inverse_cdf(&self, x: f64) -> f64 {
		if !(0.0..=1.0).contains(&x) {
			panic!("x must be in [0, 1]");
		} else {
			beta::inv_beta_reg(self.shape_a, self.shape_b, x)
		}
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		1.0
	}
}

impl Distribution for Beta {
	/// `α / (α + β)`.
	fn mean(&self) -> Option<f64> {
		Some(self.shape_a / (self.shape_a + self.shape_b))
	}

	/// The formula is `(α * β) / ((α + β)^2 * (α + β + 1))`.
	fn variance(&self) -> Option<f64> {
		Some(self.shape_a * self.shape_b
			/ ((self.shape_a + self.shape_b)
				* (self.shape_a + self.shape_b)
				* (self.shape_a + self.shape_b + 1.0)))
	}

	/// `ln(B(α, β)) - (α - 1) ψ(α) - (β - 1) ψ(β) + (α + β - 2) ψ(α + β)`,
	/// where `ψ` is the digamma function.
	fn entropy(&self) -> Option<f64> {
		Some(beta::ln_beta(self.shape_a, self.shape_b)
			- (self.shape_a - 1.0) * gamma::digamma(self.shape_a)
			- (self.shape_b - 1.0) * gamma::digamma(self.shape_b)
			+ (self.shape_a + self.shape_b - 2.0)
				* gamma::digamma(self.shape_a + self.shape_b))
	}

	/// `2(β - α) * sqrt(α + β + 1) / ((α + β + 2) * sqrt(αβ))`
	fn skewness(&self) -> Option<f64> {
		Some(2.0 * (self.shape_b - self.shape_a)
			* (self.shape_a + self.shape_b + 1.0).sqrt()
			/ ((self.shape_a + self.shape_b + 2.0)
				* (self.shape_a * self.shape_b).sqrt()))
	}
}

impl Mode<Option<f64>> for Beta {
	/// `(α - 1) / (α + β - 2)` for `α > 1` and `β > 1` or `None` otherwise.
	///
	/// Since the mode is technically only calculated for `α > 1, β > 1`,
	/// those are the only allowed values.  This constraint might be relaxed
	/// in the future.
	fn mode(&self) -> Option<f64> {
		// TODO: perhaps relax constraint in order to allow calculation
		// of anti-mode
		if self.shape_a <= 1.0 || self.shape_b <= 1.0 {
			None
		} else {
			Some((self.shape_a - 1.0)
				/ (self.shape_a + self.shape_b - 2.0))
		}
	}
}

impl Continuous for Beta {
	type T = f64;

	/// `x^(α - 1) * (1 - x)^(β - 1) / B(α, β)`, where `B` is the gamma
	/// function.
	fn pdf(&self, x: f64) -> f64 {
		if !(0.0..=1.0).contains(&x) {
			0.0
		} else if ulps_eq!(self.shape_a, 1.0)
			&& ulps_eq!(self.shape_b, 1.0)
		{
			1.0
		} else if self.shape_a > 80.0 || self.shape_b > 80.0 {
			self.ln_pdf(x).exp()
		} else {
			let bb = gamma::gamma(self.shape_a + self.shape_b)
				/ (gamma::gamma(self.shape_a)
					* gamma::gamma(self.shape_b));
			bb * x.powf(self.shape_a - 1.0)
				* (1.0 - x).powf(self.shape_b - 1.0)
		}
	}

	fn ln_pdf(&self, x: f64) -> f64 {
		if !(0.0..=1.0).contains(&x) {
			f64::NEG_INFINITY
		} else if ulps_eq!(self.shape_a, 1.0)
			&& ulps_eq!(self.shape_b, 1.0)
		{
			0.0
		} else {
			let aa = gamma::ln_gamma(self.shape_a + self.shape_b)
				- gamma::ln_gamma(self.shape_a)
				- gamma::ln_gamma(self.shape_b);
			let bb = if ulps_eq!(self.shape_a, 1.0) && x == 0.0 {
				0.0
			} else if x == 0.0 {
				f64::NEG_INFINITY
			} else {
				(self.shape_a - 1.0) * x.ln()
			};
			let cc = if ulps_eq!(self.shape_b, 1.0)
				&& ulps_eq!(x, 1.0)
			{
				0.0
			} else if ulps_eq!(x, 1.0) {
				f64::NEG_INFINITY
			} else {
				(self.shape_b - 1.0) * (1.0 - x).ln()
			};
			aa + bb + cc
		}
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Beta {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		// Generated by sampling two gamma distributions and normalizing.
		let x = super::gamma::sample_unchecked(rng, self.shape_a, 1.0);
		let y = super::gamma::sample_unchecked(rng, self.shape_b, 1.0);
		x / (x + y)
	}
}
