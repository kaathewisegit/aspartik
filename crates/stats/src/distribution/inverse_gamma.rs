use computare_special::gamma::{
	digamma, gamma, ln_gamma, regularized_lower_gamma,
	regularized_upper_gamma,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

#[cfg(feature = "python")]
use util::impl_pyerr;

#[cfg(feature = "python")]
use crate::python_macros::impl_pymethods;
use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};

/// Implements the [Inverse
/// Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
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
pub struct InverseGamma {
	shape: f64,
	rate: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for InverseGamma;
	new(shape: f64, rate: f64) -> Result<InverseGamma, InverseGammaError>;
	get(shape: f64 as py_shape);
	get(rate: f64 as py_rate);
	repr("Gamma(shape={}, rate={})", shape, rate);
	Continuous true;
	ContinuousCDF true;
	Distribution true;
}

/// Represents the errors that can occur when creating an [`InverseGamma`].
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
pub enum InverseGammaError {
	#[error("The shape is NaN, infinite, zero or less than zero")]
	ShapeInvalid,

	#[error("The rate is NaN, infinite, zero or less than zero")]
	RateInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(InverseGammaError, pyo3::exceptions::PyValueError);

impl InverseGamma {
	/// Constructs a new inverse gamma distribution with a shape (α)
	/// of `shape` and a rate (β) of `rate`
	///
	/// # Errors
	///
	/// Returns an error if `shape` or `rate` are `NaN`.
	/// Also returns an error if `shape` or `rate` are not in `(0, +inf)`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::InverseGamma;
	///
	/// let mut result = InverseGamma::new(3.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = InverseGamma::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		shape: f64,
		rate: f64,
	) -> Result<InverseGamma, InverseGammaError> {
		if shape.is_nan() || shape.is_infinite() || shape <= 0.0 {
			return Err(InverseGammaError::ShapeInvalid);
		}

		if rate.is_nan() || rate.is_infinite() || rate <= 0.0 {
			return Err(InverseGammaError::RateInvalid);
		}

		Ok(InverseGamma { shape, rate })
	}

	/// Returns the shape (α) of the inverse gamma distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::InverseGamma;
	///
	/// let n = InverseGamma::new(3.0, 1.0).unwrap();
	/// assert_eq!(n.shape(), 3.0);
	/// ```
	pub fn shape(&self) -> f64 {
		self.shape
	}

	/// Returns the rate (β) of the inverse gamma distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::InverseGamma;
	///
	/// let n = InverseGamma::new(3.0, 1.0).unwrap();
	/// assert_eq!(n.rate(), 1.0);
	/// ```
	pub fn rate(&self) -> f64 {
		self.rate
	}
}

impl core::fmt::Display for InverseGamma {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Inv-Gamma({}, {})", self.shape, self.rate)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for InverseGamma {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		1.0 / super::gamma::sample_unchecked(r, self.shape, self.rate)
	}
}

impl ContinuousCDF for InverseGamma {
	/// `Γ(α, β / x) / Γ(α)`, where the numerator is the upper incomplete
	/// gamma function, the denominator is the gamma function, `α` is the
	/// shape, and `β` is the rate
	fn cdf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			0.0
		} else if x.is_infinite() {
			1.0
		} else {
			regularized_upper_gamma(self.shape, self.rate / x)
		}
	}

	/// `Γ(α, β / x) / Γ(α)`, where the numerator is the lower incomplete
	/// gamma function, the denominator is the gamma function, `α` is the
	/// shape, and `β` is the rate
	fn sf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			1.0
		} else if x.is_infinite() {
			0.0
		} else {
			regularized_lower_gamma(self.shape, self.rate / x)
		}
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for InverseGamma {
	/// Returns the mean of the inverse distribution
	///
	/// # None
	///
	/// If `shape <= 1.0`
	///
	/// # Formula
	///
	/// ```text
	/// β / (α - 1)
	/// ```
	///
	/// where `α` is the shape and `β` is the rate
	fn mean(&self) -> Option<f64> {
		if self.shape <= 1.0 {
			None
		} else {
			Some(self.rate / (self.shape - 1.0))
		}
	}

	/// Returns the variance of the inverse gamma distribution
	///
	/// # None
	///
	/// If `shape <= 2.0`
	///
	/// # Formula
	///
	/// ```text
	/// β^2 / ((α - 1)^2 * (α - 2))
	/// ```
	///
	/// where `α` is the shape and `β` is the rate
	fn variance(&self) -> Option<f64> {
		if self.shape <= 2.0 {
			None
		} else {
			let val = self.rate * self.rate
				/ ((self.shape - 1.0)
					* (self.shape - 1.0) * (self.shape - 2.0));
			Some(val)
		}
	}

	/// Returns the entropy of the inverse gamma distribution
	///
	/// # Formula
	///
	/// ```text
	/// α + ln(β * Γ(α)) - (1 + α) * ψ(α)
	/// ```
	///
	/// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
	/// and `ψ` is the digamma function
	fn entropy(&self) -> Option<f64> {
		let entr = self.shape + self.rate.ln() + ln_gamma(self.shape)
			- (1.0 + self.shape) * digamma(self.shape);
		Some(entr)
	}

	/// Returns the skewness of the inverse gamma distribution
	///
	/// # None
	///
	/// If `shape <= 3`
	///
	/// # Formula
	///
	/// ```text
	/// 4 * sqrt(α - 2) / (α - 3)
	/// ```
	///
	/// where `α` is the shape
	fn skewness(&self) -> Option<f64> {
		if self.shape <= 3.0 {
			None
		} else {
			Some(4.0 * (self.shape - 2.0).sqrt()
				/ (self.shape - 3.0))
		}
	}
}

impl Mode<Option<f64>> for InverseGamma {
	/// Returns the mode of the inverse gamma distribution
	///
	/// # Formula
	///
	/// ```text
	/// β / (α + 1)
	/// ```
	///
	/// /// where `α` is the shape and `β` is the rate
	fn mode(&self) -> Option<f64> {
		Some(self.rate / (self.shape + 1.0))
	}
}

impl Continuous for InverseGamma {
	/// Calculates the probability density function for the
	/// inverse gamma distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (β^α / Γ(α)) * x^(-α - 1) * e^(-β / x)
	/// ```
	///
	/// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
	fn pdf(&self, x: f64) -> f64 {
		if x <= 0.0 || x.is_infinite() {
			0.0
		} else if self.shape == 1.0 {
			self.rate / (x * x) * (-self.rate / x).exp()
		} else {
			self.rate.powf(self.shape)
				* x.powf(-self.shape - 1.0) * (-self.rate / x).exp()
				/ gamma(self.shape)
		}
	}

	/// Calculates the probability density function for the
	/// inverse gamma distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((β^α / Γ(α)) * x^(-α - 1) * e^(-β / x))
	/// ```
	///
	/// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
	fn ln_pdf(&self, x: f64) -> f64 {
		self.pdf(x).ln()
	}
}
